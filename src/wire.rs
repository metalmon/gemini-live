//! Pure setup serialization + server-message/affective parsing (no I/O).
//! Populated in Tasks 3-5.

use serde_json::{json, Value};

use crate::types::SetupConfig;

/// Build the first `setup` message for a Gemini Live session. Every value is
/// sourced from `cfg`; the crate does not assemble prompts or know about
/// scenarios/gender/language directives — `cfg.system_instruction` arrives
/// fully built by the caller.
///
/// Wire shape ported verbatim from kutsu's `proto::build_setup`
/// (`src/proto.rs`), which is correct post-fixes:
/// - `enableAffectiveDialog` and `thinkingConfig` nest under
///   `generationConfig`; `proactivity` is a top-level `setup` field. Putting
///   `enableAffectiveDialog` directly under `setup` is what triggered close
///   1007 ("Unknown name enableAffectiveDialog at 'setup'") — see the
///   google-genai `_live_converters.py` converter paths.
pub fn build_setup(cfg: &SetupConfig) -> Value {
    let native = cfg.is_native();

    let mut speech = json!({
        "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": cfg.voice } }
    });
    if !native {
        if let Some(lang) = &cfg.language {
            speech["languageCode"] = json!(lang);
        }
    }

    let mut end_call = json!({
        "name": "end_call",
        "description": "Call this exactly once, at the end of the conversation, \
                        with the collected information and a final disposition.",
        "parameters": cfg.goal_schema,
    });
    if native {
        end_call["behavior"] = json!("NON_BLOCKING");
    }

    let aad = if native {
        json!({
            "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
            "prefixPaddingMs": 300,
            "silenceDurationMs": 100
        })
    } else {
        json!({
            "startOfSpeechSensitivity": "START_SENSITIVITY_LOW",
            "prefixPaddingMs": 1000
        })
    };

    let mut setup = json!({
        "model": format!("models/{}", cfg.model.model_id()),
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "temperature": cfg.temperature,
            "speechConfig": speech
        },
        "systemInstruction": { "parts": [ { "text": cfg.system_instruction } ] },
        "tools": [ { "functionDeclarations": [ end_call ] } ],
        "realtimeInputConfig": { "automaticActivityDetection": aad },
        "sessionResumption": { "handle": cfg.resume_handle },
        "inputAudioTranscription": {},
        "outputAudioTranscription": {}
    });

    if native {
        // native-audio v1alpha extras — wire placement per the google-genai
        // live converter (`_live_converters.py`): `thinkingConfig` and
        // `enableAffectiveDialog` nest under `generationConfig`; `proactivity`
        // is a top-level `setup` field.
        setup["generationConfig"]["thinkingConfig"] = json!({ "thinkingBudget": 0 });
        setup["generationConfig"]["enableAffectiveDialog"] = json!(true);
        setup["proactivity"] = json!({ "proactiveAudio": true });
    }

    json!({ "setup": setup })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Model;

    const TEMPERATURE: f32 = 0.8;

    fn cfg(model: Model, resume_handle: Option<String>) -> SetupConfig {
        SetupConfig {
            model,
            voice: "Autonoe".into(),
            language: if model.is_native() { None } else { Some("en-US".into()) },
            system_instruction: "Be nice.".into(),
            temperature: TEMPERATURE,
            goal_schema: serde_json::json!({"type":"object","required":["disposition"]}),
            resume_handle,
        }
    }

    #[test]
    fn half_cascade_setup_shape() {
        let s = build_setup(&cfg(Model::HalfCascade, None));
        let setup = &s["setup"];
        assert_eq!(setup["model"], "models/gemini-3.1-flash-live-preview");
        assert_eq!(setup["generationConfig"]["responseModalities"][0], "AUDIO");
        assert_eq!(setup["generationConfig"]["temperature"], TEMPERATURE as f64);
        assert_eq!(setup["generationConfig"]["speechConfig"]["voiceConfig"]
            ["prebuiltVoiceConfig"]["voiceName"], "Autonoe");
        assert_eq!(setup["generationConfig"]["speechConfig"]["languageCode"], "en-US");
        // Exactly one tool: end_call, parameters == goal_schema.
        let decl = &setup["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "end_call");
        assert_eq!(decl["parameters"]["required"][0], "disposition");
        assert!(decl["behavior"].is_null(), "NON_BLOCKING is native-only");
        // VAD half-cascade.
        let aad = &setup["realtimeInputConfig"]["automaticActivityDetection"];
        assert_eq!(aad["startOfSpeechSensitivity"], "START_SENSITIVITY_LOW");
        assert_eq!(aad["prefixPaddingMs"], 1000);
        assert!(aad["silenceDurationMs"].is_null());
        // Transcription both ways on.
        assert!(setup["inputAudioTranscription"].is_object());
        assert!(setup["outputAudioTranscription"].is_object());
        // No native-only fields.
        assert!(setup["enableAffectiveDialog"].is_null());
        assert!(setup["generationConfig"]["enableAffectiveDialog"].is_null());
        assert!(setup["generationConfig"]["thinkingConfig"].is_null());
        assert!(setup["proactivity"].is_null());
        // Top-level wrapper key is snake_case `setup`.
        assert!(s.get("setup").is_some());
    }

    #[test]
    fn native_audio_setup_shape() {
        let s = build_setup(&cfg(Model::NativeAudio, Some("H1".into())));
        let setup = &s["setup"];
        assert_eq!(setup["model"], "models/gemini-2.5-flash-native-audio-latest");
        // No languageCode on native.
        assert!(setup["generationConfig"]["speechConfig"]["languageCode"].is_null());
        // proactivity is a top-level setup field (path per the google-genai live
        // converter).
        assert!(setup["proactivity"]["proactiveAudio"] == true);
        // enableAffectiveDialog nests under generationConfig, NOT top-level
        // setup (top-level caused close 1007).
        assert!(setup["enableAffectiveDialog"].is_null());
        assert_eq!(setup["generationConfig"]["enableAffectiveDialog"], true);
        // Reasoning disabled on native (lower latency).
        assert_eq!(setup["generationConfig"]["thinkingConfig"]["thinkingBudget"], 0);
        // NON_BLOCKING behavior on end_call.
        assert_eq!(setup["tools"][0]["functionDeclarations"][0]["behavior"], "NON_BLOCKING");
        // Native VAD.
        let aad = &setup["realtimeInputConfig"]["automaticActivityDetection"];
        assert_eq!(aad["startOfSpeechSensitivity"], "START_SENSITIVITY_HIGH");
        assert_eq!(aad["prefixPaddingMs"], 300);
        assert_eq!(aad["silenceDurationMs"], 100);
        // Resumption handle carried.
        assert_eq!(setup["sessionResumption"]["handle"], "H1");
        // Top-level wrapper key is snake_case `setup`.
        assert!(s.get("setup").is_some());
    }

    #[test]
    fn resume_handle_none_serializes_null() {
        let s = build_setup(&cfg(Model::HalfCascade, None));
        assert!(s["setup"]["sessionResumption"]["handle"].is_null());
    }
}
