// haven_engine.cpp - llama.cpp engine for Haven Chat (CPU only).
//
// Phase 2a: load one GGUF model, chat with it (streaming), choose CPU threads.
// Embedding and reranking are added in phases 2b and 2c.
//
// All functions are called from ONE background thread in Kotlin, except
// stop(), which only sets a flag.

#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-backend.h"
#include "common.h"
#include "chat.h"

#define TAG "haven-engine"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// State (one model at a time)
// ---------------------------------------------------------------------------
static llama_model *                g_model = nullptr;
static llama_context *              g_ctx   = nullptr;
static llama_sampler *              g_smpl  = nullptr;
static common_chat_templates_ptr    g_tmpls;
static std::vector<common_chat_msg> g_msgs;    // the conversation (system, user, assistant, ...)
static std::vector<llama_token>     g_cached;  // exactly the tokens now stored in the KV cache
static std::atomic<bool>            g_stop{false};
static int                          g_n_ctx = 0;

// Last warnings/errors from llama.cpp, so the app can show them on screen.
static std::mutex       g_log_mu;
static std::string      g_log_buf;
static ggml_log_level   g_last_level = GGML_LOG_LEVEL_INFO;

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (!text) { return; }
    if (level != GGML_LOG_LEVEL_CONT) { g_last_level = level; }

    int prio = ANDROID_LOG_INFO;
    if (g_last_level == GGML_LOG_LEVEL_ERROR) { prio = ANDROID_LOG_ERROR; }
    else if (g_last_level == GGML_LOG_LEVEL_WARN) { prio = ANDROID_LOG_WARN; }
    else if (g_last_level == GGML_LOG_LEVEL_DEBUG) { prio = ANDROID_LOG_DEBUG; }
    __android_log_print(prio, TAG, "%s", text);

    if (g_last_level == GGML_LOG_LEVEL_ERROR || g_last_level == GGML_LOG_LEVEL_WARN) {
        std::lock_guard<std::mutex> lock(g_log_mu);
        g_log_buf += text;
        if (g_log_buf.size() > 6000) {
            g_log_buf.erase(0, g_log_buf.size() - 6000);
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
static std::string jstring_to_std(JNIEnv * env, jstring s) {
    if (!s) { return ""; }
    const char * c = env->GetStringUTFChars(s, nullptr);
    std::string r = c ? c : "";
    if (c) { env->ReleaseStringUTFChars(s, c); }
    return r;
}

// Prompts come from Kotlin as real UTF-8 bytes (JNI strings would break emoji).
static std::string bytes_to_std(JNIEnv * env, jbyteArray arr) {
    if (!arr) { return ""; }
    const jsize n = env->GetArrayLength(arr);
    std::string r((size_t) n, '\0');
    if (n > 0) {
        env->GetByteArrayRegion(arr, 0, n, reinterpret_cast<jbyte *>(&r[0]));
    }
    return r;
}

// Only used for short status/error texts: non-ASCII bytes become '?'.
static jstring ascii_jstring(JNIEnv * env, const std::string & s) {
    std::string t = s;
    for (auto & c : t) {
        if ((unsigned char) c >= 0x80) { c = '?'; }
    }
    return env->NewStringUTF(t.c_str());
}

// Length of the leading part of s that contains only complete UTF-8 characters.
static size_t utf8_complete_len(const std::string & s) {
    const size_t n = s.size();
    size_t i = n;
    int back = 0;
    while (i > 0 && back < 4) {
        i--;
        back++;
        const unsigned char c = (unsigned char) s[i];
        if ((c & 0xC0) != 0x80) {   // ASCII or a lead byte
            int need = 1;
            if (c >= 0xF0)      { need = 4; }
            else if (c >= 0xE0) { need = 3; }
            else if (c >= 0xC0) { need = 2; }
            return need > back ? i : n;
        }
    }
    return n;
}

static double seconds_since(const std::chrono::steady_clock::time_point & t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void reset_kv() {
    if (g_ctx) { llama_memory_clear(llama_get_memory(g_ctx), true); }
    g_cached.clear();
}

static void free_all() {
    if (g_smpl)  { llama_sampler_free(g_smpl); g_smpl = nullptr; }
    g_tmpls.reset();
    if (g_ctx)   { llama_free(g_ctx); g_ctx = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    g_msgs.clear();
    g_cached.clear();
    g_n_ctx = 0;
}

// Decodes tokens[from..] in chunks. g_cached must hold exactly tokens[0..from).
// Returns 0 = ok, 1 = stopped by the user, 2 = error.
static int decode_chunked(const std::vector<llama_token> & tokens, size_t from) {
    const size_t n_batch = std::max<size_t>(1, llama_n_batch(g_ctx));
    size_t i = from;
    while (i < tokens.size()) {
        if (g_stop) { return 1; }
        const size_t n = std::min(n_batch, tokens.size() - i);
        std::vector<llama_token> chunk(tokens.begin() + i, tokens.begin() + i + n);
        llama_batch batch = llama_batch_get_one(chunk.data(), (int32_t) n);
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("decode failed at token %zu", i);
            reset_kv();
            return 2;
        }
        g_cached.insert(g_cached.end(), chunk.begin(), chunk.end());
        i += n;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// JNI: class com.haven.chat.LlamaBridge
// ---------------------------------------------------------------------------
extern "C" {

// Call once at app start. Loads the CPU backends from the app's native lib folder.
JNIEXPORT void JNICALL
Java_com_haven_chat_LlamaBridge_init(JNIEnv * env, jobject, jstring native_lib_dir) {
    llama_log_set(log_cb, nullptr);
    const std::string dir = jstring_to_std(env, native_lib_dir);
    LOGI("loading backends from %s", dir.c_str());
    ggml_backend_load_all_from_path(dir.c_str());
    llama_backend_init();
}

// Returns 0 = ok, 1 = model file failed to load, 2 = context failed to create.
JNIEXPORT jint JNICALL
Java_com_haven_chat_LlamaBridge_loadModel(JNIEnv * env, jobject, jstring jpath, jint n_ctx, jint n_threads) {
    free_all();
    {
        std::lock_guard<std::mutex> lock(g_log_mu);
        g_log_buf.clear();
    }
    const std::string path = jstring_to_std(env, jpath);
    const int threads = std::max(1, std::min(16, (int) n_threads));

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;   // CPU only
    g_model = llama_model_load_from_file(path.c_str(), mparams);
    if (!g_model) { return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) n_ctx;
    cparams.n_batch         = 512;
    cparams.n_ubatch        = 512;
    cparams.n_threads       = threads;
    cparams.n_threads_batch = threads;
    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        return 2;
    }
    g_n_ctx = (int) llama_n_ctx(g_ctx);

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    g_smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(g_smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_temp(0.3f));
    llama_sampler_chain_add(g_smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    g_tmpls = common_chat_templates_init(g_model, "");
    g_msgs.clear();
    g_cached.clear();
    LOGI("model loaded, n_ctx = %d, threads = %d", g_n_ctx, threads);
    return 0;
}

// Change the CPU thread count without reloading the model.
JNIEXPORT void JNICALL
Java_com_haven_chat_LlamaBridge_setThreads(JNIEnv *, jobject, jint n_threads) {
    if (!g_ctx) { return; }
    const int threads = std::max(1, std::min(16, (int) n_threads));
    llama_set_n_threads(g_ctx, threads, threads);
    LOGI("threads set to %d", threads);
}

JNIEXPORT jstring JNICALL
Java_com_haven_chat_LlamaBridge_systemInfo(JNIEnv * env, jobject) {
    return ascii_jstring(env, llama_print_system_info());
}

// Start a new conversation with the given system prompt (UTF-8 bytes, may be empty).
JNIEXPORT void JNICALL
Java_com_haven_chat_LlamaBridge_resetChat(JNIEnv * env, jobject, jbyteArray jsystem) {
    g_msgs.clear();
    reset_kv();
    const std::string system = bytes_to_std(env, jsystem);
    if (!system.empty()) {
        common_chat_msg m;
        m.role    = "system";
        m.content = system;
        g_msgs.push_back(m);
    }
}

// Sends one user message and streams the answer through listener.onToken(bytes).
// Returns a one-line statistics text, or a text starting with "ERROR:".
JNIEXPORT jstring JNICALL
Java_com_haven_chat_LlamaBridge_generate(JNIEnv * env, jobject, jbyteArray jtext,
                                         jint max_new, jboolean thinking, jobject listener) {
    if (!g_ctx || !g_model || !g_smpl) {
        return env->NewStringUTF("ERROR: no model loaded");
    }
    g_stop = false;

    jclass cls = env->GetObjectClass(listener);
    jmethodID on_token = env->GetMethodID(cls, "onToken", "([B)Z");
    if (!on_token) {
        return env->NewStringUTF("ERROR: listener has no onToken method");
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    const size_t msgs_before = g_msgs.size();

    common_chat_msg user;
    user.role    = "user";
    user.content = bytes_to_std(env, jtext);
    g_msgs.push_back(user);

    // 1) Format the whole conversation with the model's own chat template.
    std::string prompt;
    try {
        common_chat_templates_inputs in;
        in.messages              = g_msgs;
        in.add_generation_prompt = true;
        in.use_jinja             = true;
        in.enable_thinking       = thinking;
        in.add_bos               = llama_vocab_get_add_bos(vocab);
        in.add_eos               = llama_vocab_get_add_eos(vocab);
        prompt = common_chat_templates_apply(g_tmpls.get(), in).prompt.str();
    } catch (const std::exception & e) {
        g_msgs.resize(msgs_before);
        return ascii_jstring(env, std::string("ERROR: chat template failed: ") + e.what());
    }

    // 2) Tokenize.
    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, /*add_special*/ true, /*parse_special*/ true);
    if (tokens.empty() || (int) tokens.size() + 16 >= g_n_ctx) {
        g_msgs.resize(msgs_before);
        char msg[200];
        snprintf(msg, sizeof(msg),
                 "ERROR: conversation too long (%d tokens, context is %d). Press New chat or choose a bigger context.",
                 (int) tokens.size(), g_n_ctx);
        return env->NewStringUTF(msg);
    }
    const int budget = std::min<int>(max_new, g_n_ctx - (int) tokens.size() - 1);

    // 3) Reuse the part of the KV cache that is identical to the new prompt.
    llama_memory_t mem = llama_get_memory(g_ctx);
    size_t n_keep = 0;
    while (n_keep < g_cached.size() && n_keep < tokens.size() && g_cached[n_keep] == tokens[n_keep]) {
        n_keep++;
    }
    if (n_keep >= tokens.size()) {
        n_keep = tokens.size() - 1;     // the last token must be decoded again to get logits
    }
    if (n_keep < g_cached.size()) {
        bool ok = n_keep > 0;
        if (ok) {
            // Sliding-window models drop old tokens; only roll back if the window is still intact.
            const int n_swa = llama_model_n_swa(g_model);
            if (n_swa > 0) {
                const llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
                if (pos_min > std::max(0, (int) n_keep - n_swa)) { ok = false; }
            }
        }
        if (ok) { ok = llama_memory_seq_rm(mem, 0, (llama_pos) n_keep, -1); }
        if (ok) {
            g_cached.resize(n_keep);
        } else {
            reset_kv();                 // safe fallback: read the whole prompt again
            n_keep = 0;
        }
    }

    // 4) Read the new part of the prompt.
    const auto t_prefill = std::chrono::steady_clock::now();
    const int rc = decode_chunked(tokens, n_keep);
    const double prefill_s = seconds_since(t_prefill);
    const size_t n_new = tokens.size() - n_keep;
    if (rc == 1) {
        g_msgs.resize(msgs_before);
        return env->NewStringUTF("STOPPED while reading the prompt");
    }
    if (rc == 2) {
        g_msgs.resize(msgs_before);
        return env->NewStringUTF("ERROR: decoding the prompt failed (see log)");
    }

    // 5) Generate.
    std::string answer;
    std::string pending;     // bytes of an unfinished UTF-8 character
    int n_gen = 0;
    bool failed = false;
    const auto t_gen = std::chrono::steady_clock::now();

    while (n_gen < budget && !g_stop) {
        const llama_token id = llama_sampler_sample(g_smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) { break; }

        pending += common_token_to_piece(vocab, id, true);
        const size_t cut = utf8_complete_len(pending);
        const std::string out = pending.substr(0, cut);
        pending.erase(0, cut);

        llama_token tok = id;
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(g_ctx, batch) != 0) {
            LOGE("decode failed while generating");
            reset_kv();
            failed = true;
            break;
        }
        g_cached.push_back(id);
        n_gen++;

        if (!out.empty()) {
            answer += out;
            jbyteArray arr = env->NewByteArray((jsize) out.size());
            env->SetByteArrayRegion(arr, 0, (jsize) out.size(), reinterpret_cast<const jbyte *>(out.data()));
            const jboolean keep_going = env->CallBooleanMethod(listener, on_token, arr);
            env->DeleteLocalRef(arr);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
                break;
            }
            if (!keep_going) { break; }
        }
    }
    const double gen_s = seconds_since(t_gen);
    const bool stopped = g_stop.load();

    // 6) Remember the answer in the conversation.
    if (!answer.empty()) {
        common_chat_msg a;
        a.role    = "assistant";
        a.content = answer;
        g_msgs.push_back(a);
    } else {
        g_msgs.resize(msgs_before);
    }

    if (failed) {
        return env->NewStringUTF("ERROR: decoding failed while generating (see log)");
    }
    char stats[256];
    snprintf(stats, sizeof(stats),
             "prompt %zu new tokens (%zu reused) %.1f t/s | answer %d tokens %.1f t/s%s",
             n_new, n_keep,
             prefill_s > 0.0 ? (double) n_new / prefill_s : 0.0,
             n_gen,
             gen_s > 0.0 ? (double) n_gen / gen_s : 0.0,
             stopped ? " | stopped" : "");
    return env->NewStringUTF(stats);
}

// Safe to call from any thread: asks the running generate() to finish.
JNIEXPORT void JNICALL
Java_com_haven_chat_LlamaBridge_stop(JNIEnv *, jobject) {
    g_stop = true;
}

JNIEXPORT void JNICALL
Java_com_haven_chat_LlamaBridge_unload(JNIEnv *, jobject) {
    free_all();
}

// Last warnings and errors printed by llama.cpp (ASCII only).
JNIEXPORT jstring JNICALL
Java_com_haven_chat_LlamaBridge_lastLog(JNIEnv * env, jobject) {
    std::lock_guard<std::mutex> lock(g_log_mu);
    return ascii_jstring(env, g_log_buf);
}

}  // extern "C"
