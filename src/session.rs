//! The reconnect/resumption session driver and the crate's public API.
//!
//! [`Session`] owns exactly one thing: the *connection lifecycle*. It connects,
//! sends `setup`, drives the transport, parses server frames into [`Event`]s,
//! remembers the session-resumption handle, and — on a resumable close —
//! reconnects with backoff and transparently resumes, all behind one unified
//! [`Session::next_event`] stream.
//!
//! It deliberately does **not** own call semantics: the greeting timer, energy
//! VAD, audio-forwarding decisions, RESUME_CUE/GREET_CUE wording, the uplink
//! drain, or `end_call` handling. Those live in the caller (kutsu), built on
//! top of this API — the caller reacts to `SessionOpened { is_reconnect }` and
//! decides what to send via [`Session::send_audio`] /
//! [`Session::send_client_text`] / [`Session::send_tool_response`].
//!
//! The reconnect/backoff policy is ported from kutsu's `gemini_live::start` +
//! `reconnect_outcome` + `reconnect::{Backoff, ReconnectState}`: backoff
//! 0.3s → ×2 → max 5s, and the stored resumption handle is dropped as stale
//! after [`HANDLE_DROP_AFTER`] consecutive failed connects.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;

use crate::transport::{ProxyConfig, Transport, TransportError, WsTransport};
use crate::types::{AffectLabel, CloseReason, Model, Role, ServerEvent, SetupConfig};
use crate::wire;

/// Drop the stored resumption handle after this many consecutive failed
/// connects (kutsu's `ReconnectState::new(4)`).
const HANDLE_DROP_AFTER: u32 = 4;

/// A produced-on-demand transport factory: each call establishes a *fresh*
/// connection (a new [`WsTransport`] in production; a scripted double in
/// tests). Used by the reconnect loop; the initial connection is established
/// eagerly by [`Session::connect`].
pub type Reconnector<T> =
    Box<dyn FnMut() -> Pin<Box<dyn Future<Output = Result<T, SessionError>> + Send>> + Send>;

/// Everything the crate needs to open (and reopen) one Gemini Live session.
/// `setup` is assembled by the caller (kutsu builds the prompt/scenario); the
/// crate only serializes and sends it.
pub struct ClientConfig {
    pub model: Model,
    pub api_key: String,
    pub proxy: Option<ProxyConfig>,
    /// The caller-assembled setup. Its `resume_handle` seeds the first connect;
    /// thereafter the crate replaces it with the latest server-issued handle.
    pub setup: SetupConfig,
    /// Give up (terminal [`Event::SessionClosed`]) after this many consecutive
    /// *non-progressing* reconnect attempts (a failed connect, or a connect
    /// that closes before `setupComplete`). `None` (the default and the match
    /// for kutsu) is unbounded: the crate reconnects forever, bounded only by
    /// the caller's own call-level hangup/time-cap. This is safe from
    /// busy-looping because the backoff caps at 5s. Set `Some(n)` only when the
    /// caller wants the crate to own a give-up tolerance.
    pub max_reconnect_attempts: Option<u32>,
}

/// One event on the unified session stream. `SessionOpened`/`SessionClosed`
/// bracket the lifecycle; the rest mirror parsed server activity. The
/// resumption handle is intentionally absent — it is managed internally and
/// never surfaced.
#[derive(Clone, Debug)]
pub enum Event {
    /// A connection is open and `setup` has been sent. `is_reconnect` is
    /// `false` for the very first open, `true` for every transparent resume —
    /// the caller's cue to drain stale uplink / send a resume cue.
    SessionOpened { is_reconnect: bool },
    /// Terminal: reconnection was exhausted or the close is unrecoverable. The
    /// stream ends after this (`next_event` returns `None` thereafter).
    SessionClosed { reason: CloseReason },
    /// 24 kHz PCM16 output audio.
    OutputAudio(Vec<i16>),
    Transcript { role: Role, text: String, final_: bool },
    Affect { role: Role, label: AffectLabel },
    Interrupted,
    TurnComplete,
    /// A tool/function call. The crate does not special-case `end_call`; the
    /// caller decides its semantics and acks via [`Session::send_tool_response`].
    ToolCall { name: String, id: String, args: serde_json::Value },
}

/// An error from establishing or driving a [`Session`].
#[derive(Debug)]
pub enum SessionError {
    Transport(TransportError),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Transport(e) => write!(f, "transport: {e}"),
        }
    }
}

impl std::error::Error for SessionError {}

impl From<TransportError> for SessionError {
    fn from(e: TransportError) -> Self {
        SessionError::Transport(e)
    }
}

/// How the just-ended attempt terminated — drives the reconnect bookkeeping
/// (kutsu's `reconnect_outcome`). `Resumable` (goAway / send / protocol error)
/// is always a failure; `RemoteClose` (a clean WS close or a dropped stream)
/// is a success only if the attempt made real progress.
#[derive(Clone, Copy)]
enum EndKind {
    Resumable,
    RemoteClose,
}

/// The reconnect/resumption driver. Generic over the transport so tests can
/// inject a scripted [`crate::transport::FakeTransport`]; [`Session::connect`]
/// pins it to the real [`WsTransport`].
pub struct Session<T: Transport = WsTransport> {
    transport: T,
    cfg: ClientConfig,
    /// The latest session-resumption handle (seeded from `cfg.setup`, updated
    /// on every `sessionResumptionUpdate`, dropped as stale per the policy).
    handle: Option<String>,
    reconnect: Reconnector<T>,
    backoff: Backoff,
    rstate: ReconnectState,
    /// Events parsed but not yet handed out (drained before any transport read).
    pending: VecDeque<Event>,
    /// Set when the current attempt must reconnect; carries the close reason to
    /// surface if reconnection turns out to be terminal.
    pending_reconnect: Option<(EndKind, CloseReason)>,
    /// Whether the current attempt reached `setupComplete` or got a fresh
    /// handle — i.e. made real progress (feeds the `RemoteClose` success rule).
    progressed: bool,
    /// Latched once a terminal `SessionClosed` has been emitted.
    terminal: bool,
}

impl Session<WsTransport> {
    /// Connect (first open) and send `setup`. The first [`Session::next_event`]
    /// yields `SessionOpened { is_reconnect: false }`.
    pub async fn connect(cfg: ClientConfig) -> Result<Self, SessionError> {
        let model = cfg.model;
        // The reconnect factory re-establishes a fresh WS each time, cloning the
        // auth/proxy config (the endpoint + key are fixed for the session).
        let reconnect: Reconnector<WsTransport> = {
            let api_key = cfg.api_key.clone();
            let proxy = cfg.proxy.clone();
            Box::new(move || {
                let api_key = api_key.clone();
                let proxy = proxy.clone();
                Box::pin(async move {
                    WsTransport::connect(model, &api_key, proxy.as_ref())
                        .await
                        .map_err(SessionError::from)
                })
            })
        };

        let transport = WsTransport::connect(model, &cfg.api_key, cfg.proxy.as_ref())
            .await
            .map_err(SessionError::from)?;
        Self::start(cfg, transport, reconnect).await
    }
}

impl<T: Transport> Session<T> {
    /// Construct a session over an already-established transport plus a
    /// reconnect factory. Public but hidden: production goes through
    /// [`Session::connect`]; this exists so tests (and kutsu's tests) can
    /// inject a scripted transport.
    #[doc(hidden)]
    pub async fn connect_with_reconnector(
        cfg: ClientConfig,
        transport: T,
        reconnect: Reconnector<T>,
    ) -> Result<Self, SessionError> {
        Self::start(cfg, transport, reconnect).await
    }

    async fn start(
        cfg: ClientConfig,
        transport: T,
        reconnect: Reconnector<T>,
    ) -> Result<Self, SessionError> {
        let handle = cfg.setup.resume_handle.clone();
        let mut s = Session {
            transport,
            cfg,
            handle,
            reconnect,
            backoff: Backoff::new(300, 5000),
            rstate: ReconnectState::new(HANDLE_DROP_AFTER),
            pending: VecDeque::new(),
            pending_reconnect: None,
            progressed: false,
            terminal: false,
        };
        s.open(false).await?;
        Ok(s)
    }

    /// Send `setup` (carrying the latest stored handle) and queue the
    /// `SessionOpened` event. Shared by the first open and every reopen.
    async fn open(&mut self, is_reconnect: bool) -> Result<(), SessionError> {
        let mut setup = self.cfg.setup.clone();
        setup.resume_handle = self.handle.clone();
        self.transport.send_text(wire::build_setup(&setup).to_string()).await?;
        self.pending.push_back(Event::SessionOpened { is_reconnect });
        Ok(())
    }

    /// The unified event stream across reconnects. Drives the transport, parses
    /// frames, resumes+backs-off internally on a resumable close, and returns
    /// `None` only after a terminal [`Event::SessionClosed`].
    ///
    /// This method *is* the reconnect loop — its future may be parked inside a
    /// backoff sleep or a reconnect handshake at any await point. Poll it from a
    /// single task and to completion; do not race it against other futures or
    /// drop/cancel it concurrently except for shutdown. Cancelling mid-reconnect
    /// loses the pending close reason and re-counts the in-flight attempt on the
    /// next call (it self-heals, but the terminal reason/attempt tally drift).
    pub async fn next_event(&mut self) -> Option<Event> {
        loop {
            if let Some(ev) = self.pending.pop_front() {
                return Some(ev);
            }
            if self.terminal {
                return None;
            }
            if let Some((kind, reason)) = self.pending_reconnect.take() {
                match self.reconnect(kind).await {
                    // `open` queued SessionOpened { is_reconnect: true }.
                    Ok(()) => continue,
                    Err(()) => {
                        self.terminal = true;
                        return Some(Event::SessionClosed { reason });
                    }
                }
            }

            match self.transport.recv().await {
                Some(Ok(bytes)) => {
                    // Malformed frames are skipped (never fatal); the crate has
                    // no Warning event and a bad frame must not kill the stream.
                    if let Ok(evs) = wire::parse_server_message(&bytes) {
                        for se in evs {
                            self.absorb(se);
                        }
                    }
                    continue;
                }
                Some(Err(TransportError::Closed(reason))) => {
                    self.pending_reconnect = Some((EndKind::RemoteClose, reason));
                    continue;
                }
                Some(Err(_other)) => {
                    // Protocol/send error: resumable, reconnect transparently.
                    self.pending_reconnect =
                        Some((EndKind::Resumable, synth_reason("transport error")));
                    continue;
                }
                None => {
                    // Stream ended without a close frame — treat as a remote close.
                    self.pending_reconnect =
                        Some((EndKind::RemoteClose, synth_reason("stream ended")));
                    continue;
                }
            }
        }
    }

    /// Map one parsed server event onto internal state or a surfaced [`Event`].
    /// The resumption handle is stored (never surfaced); `setupComplete` is
    /// internal progress; `goAway` schedules a transparent reconnect.
    fn absorb(&mut self, se: ServerEvent) {
        match se {
            ServerEvent::SetupComplete => self.progressed = true,
            ServerEvent::OutputAudio(pcm) => self.pending.push_back(Event::OutputAudio(pcm)),
            ServerEvent::Transcript { role, text, final_ } => {
                self.pending.push_back(Event::Transcript { role, text, final_ })
            }
            ServerEvent::Affect { role, label } => {
                self.pending.push_back(Event::Affect { role, label })
            }
            ServerEvent::Interrupted => self.pending.push_back(Event::Interrupted),
            ServerEvent::TurnComplete => self.pending.push_back(Event::TurnComplete),
            ServerEvent::ToolCall { name, id, args } => {
                self.pending.push_back(Event::ToolCall { name, id, args })
            }
            ServerEvent::ResumptionHandle(h) => {
                self.handle = Some(h);
                self.progressed = true;
            }
            // Don't surface goAway; finish draining this frame's events, then
            // reconnect transparently.
            ServerEvent::GoAway => {
                self.pending_reconnect = Some((EndKind::Resumable, synth_reason("goAway")));
            }
        }
    }

    /// Record one non-progressing reconnect attempt (a failed connect, or a
    /// connect that closed before `setupComplete`): advance the consecutive-
    /// failure count, drop the stale handle at the threshold, and report whether
    /// the configured attempt budget is now exhausted (→ terminal).
    fn note_failure(&mut self) -> bool {
        if self.rstate.on_failure() {
            self.handle = None; // stale handle after N consecutive failures
        }
        matches!(self.cfg.max_reconnect_attempts, Some(max) if self.rstate.fails() >= max)
    }

    /// Perform the reconnect for a just-ended attempt: apply the
    /// `reconnect_outcome` bookkeeping, sleep the backoff, then (re)connect —
    /// retrying failed connects with escalating backoff and stale-handle drop —
    /// until a fresh transport is open + `setup` re-sent, or the attempt budget
    /// is exhausted (terminal).
    ///
    /// Crucially, the backoff sleep happens **before every reconnect**, not only
    /// after a failed connect. An endpoint that accepts the WS + setup and then
    /// closes before `setupComplete` (post-handshake reject / overload / rapid
    /// goAway) makes `connect` *succeed* every time; sleeping only on connect
    /// failure would hot-loop that accept-then-close forever. Sleeping up front
    /// (with backoff advancing — it is reset only on a *progressing* close, per
    /// `reconnect_outcome`'s `RemoteClose => Some(progressed)`) backs the storm
    /// off exactly like kutsu's sleep-before-every-reconnect.
    async fn reconnect(&mut self, end: EndKind) -> Result<(), ()> {
        // Bookkeeping for the attempt that just ended (kutsu's reconnect_outcome):
        // Resumable is always a failure; RemoteClose is success iff progress was
        // made — otherwise a consistently-broken endpoint keeps escalating.
        let success = matches!(end, EndKind::RemoteClose) && self.progressed;
        self.progressed = false;
        if success {
            self.rstate.on_success();
            self.backoff.reset();
        } else if self.note_failure() {
            return Err(());
        }

        loop {
            // Always back off before (re)connecting — see the method doc: this
            // is what tames an accept-then-close storm, where `connect` keeps
            // succeeding but the session never progresses.
            tokio::time::sleep(self.backoff.next_delay()).await;

            if let Ok(t) = (self.reconnect)().await {
                self.transport = t;
                if self.open(true).await.is_ok() {
                    return Ok(());
                }
                // setup send failed on the fresh transport: treat as a failed
                // attempt and keep retrying.
            }
            if self.note_failure() {
                return Err(());
            }
        }
    }

    /// Send one uplink audio frame (PCM16 @ 16 kHz). Bytes are byte-identical
    /// to kutsu's `build_realtime_input`.
    pub async fn send_audio(&mut self, pcm16_16k: &[i16]) -> Result<(), SessionError> {
        self.transport.send_text(wire::build_realtime_input(pcm16_16k)).await.map_err(Into::into)
    }

    /// Send a client text turn (kutsu uses this for GREET_CUE / RESUME_CUE).
    pub async fn send_client_text(&mut self, text: &str) -> Result<(), SessionError> {
        self.transport.send_text(wire::build_client_content(text)).await.map_err(Into::into)
    }

    /// Acknowledge a tool call by id (kutsu owns the tool's semantics).
    pub async fn send_tool_response(&mut self, call_id: &str) -> Result<(), SessionError> {
        self.transport.send_text(wire::build_tool_response(call_id)).await.map_err(Into::into)
    }
}

/// A synthetic close reason for ends that carry no server close frame.
fn synth_reason(reason: &str) -> CloseReason {
    CloseReason { code: 0, reason: reason.to_string() }
}

// --- Pure reconnect policy (ported from kutsu's `src/reconnect.rs`). ---------

/// Exponential backoff: `base` → ×2 → capped at `max`.
struct Backoff {
    current_ms: u64,
    base_ms: u64,
    max_ms: u64,
}

impl Backoff {
    fn new(base_ms: u64, max_ms: u64) -> Self {
        Backoff { current_ms: base_ms, base_ms, max_ms }
    }

    fn next_delay(&mut self) -> std::time::Duration {
        let d = std::time::Duration::from_millis(self.current_ms);
        self.current_ms = (self.current_ms * 2).min(self.max_ms);
        d
    }

    fn reset(&mut self) {
        self.current_ms = self.base_ms;
    }
}

/// Consecutive-failure counter that signals when the (likely stale) resumption
/// handle should be dropped.
struct ReconnectState {
    fails: u32,
    reset_handle_after: u32,
}

impl ReconnectState {
    fn new(reset_handle_after: u32) -> Self {
        ReconnectState { fails: 0, reset_handle_after }
    }

    fn on_success(&mut self) {
        self.fails = 0;
    }

    /// Record a failure; returns true when the caller should drop the handle.
    fn on_failure(&mut self) -> bool {
        self.fails += 1;
        self.reset_handle_after != 0 && self.fails >= self.reset_handle_after
    }

    /// The current consecutive-failure count (drives the terminal attempt bound).
    fn fails(&self) -> u32 {
        self.fails
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::FakeTransport;

    fn cfg() -> ClientConfig {
        ClientConfig {
            model: Model::HalfCascade,
            api_key: "K".into(),
            proxy: None,
            setup: SetupConfig {
                model: Model::HalfCascade,
                voice: "Autonoe".into(),
                language: Some("en-US".into()),
                system_instruction: "Be nice.".into(),
                temperature: 0.8,
                goal_schema: serde_json::json!({ "type": "object" }),
                resume_handle: None,
            },
            // Matches kutsu: unbounded reconnect (backoff caps at 5s). Tests
            // that need deterministic termination override with `Some(n)`.
            max_reconnect_attempts: None,
        }
    }

    /// A reconnector that always fails (unused by tests that never reconnect,
    /// or used to drive the terminal path).
    fn no_reconnect() -> Reconnector<FakeTransport> {
        Box::new(|| {
            Box::pin(async {
                Err(SessionError::Transport(TransportError::Connect("no reconnect".into())))
            })
        })
    }

    /// A reconnector that hands back `next` exactly once, then fails.
    fn once(next: FakeTransport) -> Reconnector<FakeTransport> {
        let mut slot = Some(next);
        Box::new(move || {
            let t = slot.take();
            Box::pin(async move {
                t.ok_or(SessionError::Transport(TransportError::Connect("drained".into())))
            })
        })
    }

    /// A reconnector that yields `n` accept-then-close transports (each accepts
    /// `setup`, then closes before any `setupComplete` — i.e. no progress),
    /// then fails. Models a post-handshake-reject / overload storm.
    fn accept_then_close(n: usize) -> Reconnector<FakeTransport> {
        let mut queue: VecDeque<FakeTransport> = (0..n)
            .map(|_| {
                let mut t = FakeTransport::new(false);
                t.push_close(1013, "overloaded");
                t
            })
            .collect();
        Box::new(move || {
            let t = queue.pop_front();
            Box::pin(async move {
                t.ok_or(SessionError::Transport(TransportError::Connect("drained".into())))
            })
        })
    }

    #[tokio::test]
    async fn first_open_emits_session_opened_and_sends_setup() {
        let fake = FakeTransport::new(true);
        let sent = fake.sent.clone();
        let mut s = Session::connect_with_reconnector(cfg(), fake, no_reconnect()).await.unwrap();

        let ev = s.next_event().await;
        assert!(matches!(ev, Some(Event::SessionOpened { is_reconnect: false })));
        // setup was captured on the fake as the first outgoing frame.
        assert!(sent.lock().unwrap()[0].contains("\"setup\""));
    }

    #[tokio::test]
    async fn events_surface_in_order_from_scripted_frames() {
        // base64 "AAABAAIAAwA=" -> i16 LE [0,1,2,3].
        let mut fake = FakeTransport::new(true);
        fake.push_data(br#"{"setupComplete":{}}"#.to_vec());
        fake.push_data(
            br#"{"serverContent":{"modelTurn":{"parts":[{"inlineData":{"data":"AAABAAIAAwA="}}]}}}"#
                .to_vec(),
        );
        fake.push_data(
            br#"{"serverContent":{"outputTranscription":{"text":"Hi","finished":true}}}"#.to_vec(),
        );
        fake.push_data(
            br#"{"toolCall":{"functionCalls":[{"name":"end_call","id":"c1","args":{"d":"x"}}]}}"#
                .to_vec(),
        );
        fake.push_data(br#"{"serverContent":{"turnComplete":true}}"#.to_vec());
        fake.push_data(br#"{"serverContent":{"interrupted":true}}"#.to_vec());

        let mut s = Session::connect_with_reconnector(cfg(), fake, no_reconnect()).await.unwrap();

        assert!(matches!(s.next_event().await, Some(Event::SessionOpened { is_reconnect: false })));
        // setupComplete surfaces nothing; audio is next.
        assert!(matches!(s.next_event().await, Some(Event::OutputAudio(a)) if a == vec![0, 1, 2, 3]));
        assert!(matches!(
            s.next_event().await,
            Some(Event::Transcript { role: Role::Model, text, final_: true }) if text == "Hi"
        ));
        assert!(matches!(
            s.next_event().await,
            Some(Event::ToolCall { name, id, .. }) if name == "end_call" && id == "c1"
        ));
        assert!(matches!(s.next_event().await, Some(Event::TurnComplete)));
        assert!(matches!(s.next_event().await, Some(Event::Interrupted)));
    }

    #[tokio::test(start_paused = true)]
    async fn resumption_handle_stored_not_surfaced_then_replayed_on_reconnect() {
        // Script 1: setupComplete, a fresh handle, then a resumable close.
        let mut first = FakeTransport::new(false);
        first.push_data(br#"{"setupComplete":{}}"#.to_vec());
        first.push_data(br#"{"sessionResumptionUpdate":{"newHandle":"H1"}}"#.to_vec());
        first.push_close(1011, "server restart");

        // Script 2: the reopened connection. Its captured `sent` must show the
        // setup carrying the stored handle "H1".
        let second = FakeTransport::new(true);
        let second_sent = second.sent.clone();

        let mut s =
            Session::connect_with_reconnector(cfg(), first, once(second)).await.unwrap();

        // First open.
        assert!(matches!(s.next_event().await, Some(Event::SessionOpened { is_reconnect: false })));
        // The handle update surfaces nothing on its own; the next event is the
        // transparent reopen after the scripted close.
        assert!(matches!(s.next_event().await, Some(Event::SessionOpened { is_reconnect: true })));

        // The reopen re-sent setup carrying the stored handle.
        let setup = &second_sent.lock().unwrap()[0];
        assert!(setup.contains("\"setup\""));
        assert!(setup.contains("H1"), "reopened setup must carry the stored handle: {setup}");
    }

    #[tokio::test]
    async fn send_audio_is_byte_identical_to_kutsu() {
        let fake = FakeTransport::new(true);
        let sent = fake.sent.clone();
        let mut s = Session::connect_with_reconnector(cfg(), fake, no_reconnect()).await.unwrap();
        let _ = s.next_event().await; // SessionOpened

        s.send_audio(&[0i16, 1i16]).await.unwrap();

        assert_eq!(
            sent.lock().unwrap().last().unwrap(),
            r#"{"realtime_input":{"audio":{"data":"AAABAA==","mimeType":"audio/pcm;rate=16000"}}}"#
        );
    }

    #[tokio::test(start_paused = true)]
    async fn terminal_close_emits_session_closed_then_none() {
        let mut fake = FakeTransport::new(false);
        fake.push_data(br#"{"setupComplete":{}}"#.to_vec());
        fake.push_close(1011, "boom");

        // Bound reconnection so it terminates deterministically: one failed
        // connect is enough to exhaust the budget.
        let mut c = cfg();
        c.max_reconnect_attempts = Some(1);
        // No reconnector transport available -> reconnection is exhausted.
        let mut s = Session::connect_with_reconnector(c, fake, no_reconnect()).await.unwrap();

        assert!(matches!(s.next_event().await, Some(Event::SessionOpened { is_reconnect: false })));
        // Drives: setupComplete -> close -> failed reconnects -> terminal.
        match s.next_event().await {
            Some(Event::SessionClosed { reason }) => assert_eq!(reason.code, 1011),
            other => panic!("expected SessionClosed, got {other:?}"),
        }
        // Stream has ended.
        assert!(s.next_event().await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn accept_then_close_storm_backs_off_and_terminates_at_the_bound() {
        // Every connection accepts setup then closes before `setupComplete` —
        // no progress. This must NOT hot-loop: each reconnect backs off (the
        // backoff advances because a no-progress close never resets it), and
        // with a bound it terminates after exactly `n` non-progressing attempts.
        let mut initial = FakeTransport::new(false);
        initial.push_close(1013, "overloaded");

        let mut c = cfg();
        c.max_reconnect_attempts = Some(3);
        // n-1 reconnect transports: initial + 2 reopens = 3 no-progress closes.
        let mut s =
            Session::connect_with_reconnector(c, initial, accept_then_close(2)).await.unwrap();

        let start = tokio::time::Instant::now();

        // First open, then two transparent reopens (each accept-then-close),
        // then the bound is hit → terminal SessionClosed → None.
        assert!(matches!(s.next_event().await, Some(Event::SessionOpened { is_reconnect: false })));
        assert!(matches!(s.next_event().await, Some(Event::SessionOpened { is_reconnect: true })));
        assert!(matches!(s.next_event().await, Some(Event::SessionOpened { is_reconnect: true })));
        match s.next_event().await {
            Some(Event::SessionClosed { reason }) => assert_eq!(reason.code, 1013),
            other => panic!("expected SessionClosed after the bound, got {other:?}"),
        }
        assert!(s.next_event().await.is_none());

        // Backoff advanced across reopens (300ms + 600ms), not a zero-delay
        // storm and not a constant base delay (which would total 600ms).
        assert!(
            start.elapsed() >= std::time::Duration::from_millis(800),
            "reconnects must back off with an advancing delay, elapsed = {:?}",
            start.elapsed()
        );
    }

    #[test]
    fn backoff_doubles_and_caps_then_resets() {
        let mut b = Backoff::new(300, 5000);
        assert_eq!(b.next_delay(), std::time::Duration::from_millis(300));
        assert_eq!(b.next_delay(), std::time::Duration::from_millis(600));
        assert_eq!(b.next_delay(), std::time::Duration::from_millis(1200));
        assert_eq!(b.next_delay(), std::time::Duration::from_millis(2400));
        assert_eq!(b.next_delay(), std::time::Duration::from_millis(4800));
        assert_eq!(b.next_delay(), std::time::Duration::from_millis(5000)); // capped
        b.reset();
        assert_eq!(b.next_delay(), std::time::Duration::from_millis(300));
    }

    #[test]
    fn reconnect_state_drops_handle_after_n_failures() {
        let mut s = ReconnectState::new(HANDLE_DROP_AFTER);
        assert!(!s.on_failure());
        assert!(!s.on_failure());
        assert!(!s.on_failure());
        assert!(s.on_failure()); // 4th -> drop
        s.on_success();
        assert!(!s.on_failure()); // reset by success
    }
}
