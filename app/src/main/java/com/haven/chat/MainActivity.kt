package com.haven.chat

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.text.InputType
import android.text.SpannableString
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import java.io.File
import java.time.LocalDate
import java.util.concurrent.Executors

/**
 * Phase 2a: engine test screen. Load Gemma from a file, chat, see tokens/s.
 * All native calls run on one background thread (worker).
 */
class MainActivity : Activity() {

    // Created on the worker thread, so a native-library problem shows a message instead of crashing.
    @Volatile private var bridge: LlamaBridge? = null
    private val worker = Executors.newSingleThreadExecutor()

    private var engineReady = false
    private var loaded = false
    private var busy = false

    private lateinit var toggleBtn: Button
    private lateinit var statusText: TextView
    private lateinit var panel: LinearLayout
    private lateinit var permText: TextView
    private lateinit var permBtn: Button
    private lateinit var pathEdit: EditText
    private lateinit var threadsSpinner: Spinner
    private lateinit var ctxSpinner: Spinner
    private lateinit var thinkingBox: CheckBox
    private lateinit var showThinkBox: CheckBox
    private lateinit var systemEdit: EditText
    private lateinit var loadBtn: Button
    private lateinit var newChatBtn: Button
    private lateinit var embedPathEdit: EditText
    private lateinit var rerankPathEdit: EditText
    private lateinit var selfTestBtn: Button
    private lateinit var dbTestBtn: Button
    private lateinit var searchTestBtn: Button
    private lateinit var panelHolder: ScrollView
    private lateinit var infoText: TextView
    private lateinit var scroll: ScrollView
    private lateinit var outputText: TextView
    private lateinit var inputEdit: EditText
    private lateinit var sendBtn: Button
    private lateinit var stopBtn: Button

    private val ctxChoices = listOf(2048, 4096, 8192)

    // Colors for the chat text
    private val answerColor = Color.parseColor("#1F1F1F")
    private val thinkingColor = Color.parseColor("#8A8A8A")

    // Splits Gemma's stream into thinking and answer (UI thread only)
    private var splitter: ThinkingSplitter? = null

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    // ------------------------------------------------------------------ setup

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(buildUi())
        updatePermText()

        worker.execute {
            try {
                val b = LlamaBridge()
                b.init(applicationInfo.nativeLibraryDir)
                val info = b.systemInfo().trim()
                bridge = b
                runOnUiThread {
                    engineReady = true
                    infoText.text = "CPU: " + info.take(260)
                    statusText.text = "Engine ready. Load a model."
                }
            } catch (t: Throwable) {
                runOnUiThread {
                    statusText.text = "Engine failed to start: " + t.javaClass.simpleName + " " + t.message
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        updatePermText()
    }

    override fun onDestroy() {
        super.onDestroy()
        bridge?.stop()
        worker.execute {
            try {
                bridge?.unload()
            } catch (t: Throwable) {
                // closing anyway
            }
        }
        worker.shutdown()
    }

    // ------------------------------------------------------------------ UI

    private fun label(caption: String): TextView {
        val tv = TextView(this)
        tv.text = caption
        tv.textSize = 13f
        tv.setPadding(0, dp(6), dp(6), 0)
        return tv
    }

    private fun buildUi(): View {
        val root = LinearLayout(this)
        root.orientation = LinearLayout.VERTICAL
        root.setPadding(dp(12), dp(8), dp(12), dp(8))

        // Top bar: settings toggle + status
        val top = LinearLayout(this)
        top.orientation = LinearLayout.HORIZONTAL
        top.gravity = Gravity.CENTER_VERTICAL
        toggleBtn = Button(this)
        toggleBtn.isAllCaps = false
        toggleBtn.text = "Hide settings"
        statusText = TextView(this)
        statusText.textSize = 13f
        statusText.text = "Starting engine..."
        top.addView(toggleBtn)
        top.addView(statusText, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(top)

        // Settings panel
        panel = LinearLayout(this)
        panel.orientation = LinearLayout.VERTICAL

        val permRow = LinearLayout(this)
        permRow.orientation = LinearLayout.HORIZONTAL
        permRow.gravity = Gravity.CENTER_VERTICAL
        permText = TextView(this)
        permText.textSize = 13f
        permBtn = Button(this)
        permBtn.isAllCaps = false
        permBtn.text = "Allow file access"
        permRow.addView(permText, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        permRow.addView(permBtn)
        panel.addView(permRow)

        panel.addView(label("Model file (put models in /storage/emulated/0/HavenModels)"))
        pathEdit = EditText(this)
        pathEdit.setSingleLine(true)
        pathEdit.textSize = 13f
        pathEdit.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        pathEdit.setText("/storage/emulated/0/HavenModels/gemma4-e2b-q4_0.gguf")
        panel.addView(pathEdit)

        val choiceRow = LinearLayout(this)
        choiceRow.orientation = LinearLayout.HORIZONTAL
        choiceRow.gravity = Gravity.CENTER_VERTICAL
        threadsSpinner = Spinner(this)
        threadsSpinner.adapter = makeAdapter((1..6).map { it.toString() })
        threadsSpinner.setSelection(3)   // 4 threads
        ctxSpinner = Spinner(this)
        ctxSpinner.adapter = makeAdapter(ctxChoices.map { it.toString() })
        ctxSpinner.setSelection(1)       // 4096
        choiceRow.addView(label("Threads"))
        choiceRow.addView(threadsSpinner)
        choiceRow.addView(label("  Context"))
        choiceRow.addView(ctxSpinner)
        panel.addView(choiceRow)

        thinkingBox = CheckBox(this)
        thinkingBox.text = "Let Gemma think first (slower)"
        thinkingBox.textSize = 13f
        panel.addView(thinkingBox)

        showThinkBox = CheckBox(this)
        showThinkBox.text = "Show the thinking text in the chat"
        showThinkBox.textSize = 13f
        showThinkBox.isChecked = true
        panel.addView(showThinkBox)

        panel.addView(label("System prompt (used on Load and New chat)"))
        systemEdit = EditText(this)
        systemEdit.textSize = 13f
        systemEdit.minLines = 2
        systemEdit.maxLines = 4
        systemEdit.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
        systemEdit.setText("You are a helpful assistant. Answer clearly and briefly. Plain text only, no Markdown symbols.")
        panel.addView(systemEdit)

        val btnRow = LinearLayout(this)
        btnRow.orientation = LinearLayout.HORIZONTAL
        loadBtn = Button(this)
        loadBtn.isAllCaps = false
        loadBtn.text = "Load model"
        newChatBtn = Button(this)
        newChatBtn.isAllCaps = false
        newChatBtn.text = "New chat"
        btnRow.addView(loadBtn)
        btnRow.addView(newChatBtn)
        panel.addView(btnRow)

        panel.addView(label("Search models (used by the self-test)"))
        embedPathEdit = EditText(this)
        embedPathEdit.setSingleLine(true)
        embedPathEdit.textSize = 13f
        embedPathEdit.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        embedPathEdit.setText("/storage/emulated/0/HavenModels/bge-m3-F16.gguf")
        panel.addView(embedPathEdit)
        rerankPathEdit = EditText(this)
        rerankPathEdit.setSingleLine(true)
        rerankPathEdit.textSize = 13f
        rerankPathEdit.inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        rerankPathEdit.setText("/storage/emulated/0/HavenModels/Bge-M3-568M-F16.gguf")
        panel.addView(rerankPathEdit)
        selfTestBtn = Button(this)
        selfTestBtn.isAllCaps = false
        selfTestBtn.text = "Search models self-test"
        panel.addView(selfTestBtn)

        dbTestBtn = Button(this)
        dbTestBtn.isAllCaps = false
        dbTestBtn.text = "Database self-test"
        panel.addView(dbTestBtn)

        searchTestBtn = Button(this)
        searchTestBtn.isAllCaps = false
        searchTestBtn.text = "Search pipeline test (unloads Gemma)"
        panel.addView(searchTestBtn)

        infoText = TextView(this)
        infoText.textSize = 11f
        panel.addView(infoText)

        panelHolder = ScrollView(this)
        panelHolder.addView(panel)
        root.addView(panelHolder, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        // Output
        scroll = ScrollView(this)
        outputText = TextView(this)
        outputText.textSize = 15f
        outputText.setTextColor(answerColor)
        outputText.setPadding(dp(4), dp(8), dp(4), dp(8))
        outputText.setTextIsSelectable(true)
        outputText.text = "Load the model, then type a message below."
        scroll.addView(outputText)
        root.addView(scroll, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))

        // Input row
        val inputRow = LinearLayout(this)
        inputRow.orientation = LinearLayout.HORIZONTAL
        inputRow.gravity = Gravity.CENTER_VERTICAL
        inputEdit = EditText(this)
        inputEdit.hint = "Message"
        inputEdit.maxLines = 4
        inputEdit.inputType = InputType.TYPE_CLASS_TEXT or
            InputType.TYPE_TEXT_FLAG_MULTI_LINE or
            InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
        sendBtn = Button(this)
        sendBtn.isAllCaps = false
        sendBtn.text = "Send"
        stopBtn = Button(this)
        stopBtn.isAllCaps = false
        stopBtn.text = "Stop"
        stopBtn.isEnabled = false
        inputRow.addView(inputEdit, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        inputRow.addView(sendBtn)
        inputRow.addView(stopBtn)
        root.addView(inputRow)

        // Listeners
        toggleBtn.setOnClickListener {
            panelHolder.visibility = if (panelHolder.visibility == View.VISIBLE) View.GONE else View.VISIBLE
            updateToggleText()
        }
        permBtn.setOnClickListener { askAllFilesAccess() }
        loadBtn.setOnClickListener { onLoadClicked() }
        newChatBtn.setOnClickListener { onNewChatClicked() }
        selfTestBtn.setOnClickListener { onSelfTestClicked() }
        dbTestBtn.setOnClickListener { onDbTestClicked() }
        searchTestBtn.setOnClickListener { onSearchTestClicked() }
        sendBtn.setOnClickListener { onSendClicked() }
        stopBtn.setOnClickListener { bridge?.stop() }

        threadsSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                val n = position + 1
                if (loaded && !busy) {
                    worker.execute { bridge?.setThreads(n) }
                    statusText.text = "CPU threads: $n"
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
        ctxSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (loaded && !busy) {
                    statusText.text = "Context changes after you press Load model again"
                }
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }

        return root
    }

    private fun makeAdapter(items: List<String>): ArrayAdapter<String> {
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, items)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        return adapter
    }

    private fun updateToggleText() {
        toggleBtn.text = if (panelHolder.visibility == View.VISIBLE) "Hide settings" else "Settings"
    }

    private fun setBusy(b: Boolean) {
        busy = b
        loadBtn.isEnabled = !b
        newChatBtn.isEnabled = !b
        selfTestBtn.isEnabled = !b
        dbTestBtn.isEnabled = !b
        searchTestBtn.isEnabled = !b
        sendBtn.isEnabled = !b
        stopBtn.isEnabled = b
    }

    private fun appendOutput(s: String) {
        outputText.append(s)
        scroll.post { scroll.scrollTo(0, outputText.height) }
    }

    private fun appendStyled(s: String, color: Int, size: Float) {
        val styled = SpannableString(s)
        styled.setSpan(ForegroundColorSpan(color), 0, styled.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        styled.setSpan(RelativeSizeSpan(size), 0, styled.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        outputText.append(styled)
        scroll.post { scroll.scrollTo(0, outputText.height) }
    }

    /** Thinking goes in small grey text, the answer in normal text. */
    private fun newSplitter(showThinking: Boolean): ThinkingSplitter {
        return ThinkingSplitter(
            onThinkingStart = {
                if (showThinking) {
                    appendStyled("Thinking:\n", thinkingColor, 0.85f)
                } else {
                    appendStyled("Thinking...", thinkingColor, 0.85f)
                }
            },
            onThinking = { text ->
                if (showThinking) appendStyled(text, thinkingColor, 0.85f)
            },
            onThinkingEnd = { appendOutput("\n\n") },
            onAnswer = { text -> appendOutput(text) }
        )
    }

    private fun toast(msg: String) {
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
    }

    // ------------------------------------------------------------------ file access

    private fun hasAllFilesAccess(): Boolean {
        return if (Build.VERSION.SDK_INT >= 30) Environment.isExternalStorageManager() else true
    }

    private fun updatePermText() {
        if (hasAllFilesAccess()) {
            permText.text = "File access: allowed"
            permBtn.visibility = View.GONE
        } else {
            permText.text = "File access: NOT allowed"
            permBtn.visibility = View.VISIBLE
        }
    }

    private fun askAllFilesAccess() {
        if (Build.VERSION.SDK_INT < 30) {
            return
        }
        try {
            startActivity(
                Intent(
                    Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    Uri.parse("package:$packageName")
                )
            )
        } catch (e: Exception) {
            startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
        }
    }

    // ------------------------------------------------------------------ actions

    private fun onLoadClicked() {
        if (busy) return
        if (!engineReady) {
            toast("The engine is not ready yet")
            return
        }
        if (!hasAllFilesAccess()) {
            toast("Tap 'Allow file access' first")
            return
        }
        val path = pathEdit.text.toString().trim()
        if (!File(path).isFile) {
            statusText.text = "File not found: $path"
            return
        }
        val threads = threadsSpinner.selectedItemPosition + 1
        val ctx = ctxChoices[ctxSpinner.selectedItemPosition]
        val systemBytes = systemEdit.text.toString().toByteArray(Charsets.UTF_8)

        loaded = false
        setBusy(true)
        statusText.text = "Loading model... (this can take a minute)"

        worker.execute {
            val b = bridge
            val t0 = System.currentTimeMillis()
            var rc = -1
            if (b != null) {
                try {
                    rc = b.loadModel(path, ctx, threads)
                    if (rc == 0) b.resetChat(systemBytes)
                } catch (t: Throwable) {
                    rc = -1
                }
            }
            val secs = (System.currentTimeMillis() - t0) / 1000.0
            val log = if (rc != 0 && b != null) b.lastLog() else ""
            runOnUiThread {
                loaded = (rc == 0)
                if (rc == 0) {
                    statusText.text = "Loaded in " + "%.1f".format(secs) + " s | $threads threads | context $ctx"
                    outputText.text = ""
                    panelHolder.visibility = View.GONE
                    updateToggleText()
                } else {
                    statusText.text = "Load failed (code $rc). " + log.takeLast(500)
                }
                setBusy(false)
            }
        }
    }

    // ------------------------------------------------------------------ search models self-test

    private class TestCase(val query: String, val answer: Int)   // answer = index of the right memory

    private val testMemories = listOf(
        "I am building an Android app called Haven Chat that runs an AI model on my phone with llama.cpp.",
        "My 4-bit counter circuit had a timing race. Adding a pull-down resistor on the clear pin of the 74HC161 fixed the clock skew.",
        "I decided to use SQLite with FTS5 for keyword search and keep all vectors in the same database file.",
        "My phone is a Realme P3 with 8 GB of RAM, so I can only load one model at a time.",
        "For lunch on Friday I plan to cook paneer butter masala with garlic naan."
    )

    private val testCases = listOf(
        TestCase("How did I fix the glitch in my digital logic project?", 1),
        TestCase("Which database did I choose for searching old chats?", 2),
        TestCase("How much memory does my handset have?", 3),
        TestCase("What am I planning to eat on Friday?", 4),
        TestCase("What is the name of the mobile application I am developing?", 0)
    )

    private fun say(line: String) {
        runOnUiThread { appendStyled(line + "\n", Color.parseColor("#444444"), 0.9f) }
    }

    private fun secs(t0: Long): String = "%.1f".format((System.nanoTime() - t0) / 1e9)

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) sum += a[i] * b[i]
        return sum
    }

    private fun loadError(what: String, rc: Int, b: LlamaBridge): String {
        val reason = when (rc) {
            1 -> "the $what file could not be loaded"
            2 -> "the $what context could not be created (not enough RAM?)"
            4 -> "the main model is still loaded"
            else -> "error code $rc"
        }
        return "  FAILED: $reason. " + b.lastLog().takeLast(300)
    }

    private fun onSelfTestClicked() {
        if (busy) return
        if (!engineReady) {
            toast("The engine is not ready yet")
            return
        }
        if (!hasAllFilesAccess()) {
            toast("Tap 'Allow file access' first")
            return
        }
        val embPath = embedPathEdit.text.toString().trim()
        val rerPath = rerankPathEdit.text.toString().trim()
        if (!File(embPath).isFile) {
            statusText.text = "File not found: $embPath"
            return
        }
        if (!File(rerPath).isFile) {
            statusText.text = "File not found: $rerPath"
            return
        }
        val threads = threadsSpinner.selectedItemPosition + 1

        loaded = false          // the main model is unloaded by the test
        setBusy(true)
        panelHolder.visibility = View.GONE
        updateToggleText()
        statusText.text = "Self-test running..."
        appendOutput("\n\n")
        say("=== Search models self-test ===")

        worker.execute {
            val b = bridge
            try {
                if (b == null) {
                    say("The engine is not available.")
                } else {
                    runSelfTest(b, embPath, rerPath, threads)
                }
            } catch (t: Throwable) {
                say("Self-test crashed: " + t.javaClass.simpleName + " " + t.message)
            }
            runOnUiThread {
                statusText.text = "Self-test finished. Press Load model to chat again."
                setBusy(false)
            }
        }
    }

    // Runs on the worker thread.
    private fun runSelfTest(b: LlamaBridge, embPath: String, rerPath: String, threads: Int) {
        val utf8 = Charsets.UTF_8

        say("Step 1: free the main model (RAM)")
        b.unload()

        // ---- embedder
        say("Step 2: embedder (bge-m3)")
        var t0 = System.nanoTime()
        var rc = b.loadAux(embPath, LlamaBridge.KIND_EMBEDDER, threads)
        if (rc != 0) {
            say(loadError("embedder", rc, b))
            return
        }
        say("  loaded in " + secs(t0) + " s | " + b.auxInfo())

        t0 = System.nanoTime()
        val memVecs = ArrayList<FloatArray>()
        for (m in testMemories) {
            val v = b.embed(m.toByteArray(utf8))
            if (v == null) {
                say("  FAILED to embed a memory. " + b.lastLog().takeLast(300))
                b.unloadAux()
                return
            }
            memVecs.add(v)
        }
        val qVecs = ArrayList<FloatArray>()
        for (c in testCases) {
            val v = b.embed(c.query.toByteArray(utf8))
            if (v == null) {
                say("  FAILED to embed a question. " + b.lastLog().takeLast(300))
                b.unloadAux()
                return
            }
            qVecs.add(v)
        }
        val nTexts = memVecs.size + qVecs.size
        val embSec = (System.nanoTime() - t0) / 1e9
        say("  embedded " + nTexts + " texts in " + "%.1f".format(embSec) + " s (vector size " + memVecs[0].size + ")")
        b.unloadAux()

        // ---- vector search (top 3 per question)
        val top = ArrayList<List<Int>>()
        for (qi in testCases.indices) {
            top.add(memVecs.indices.sortedByDescending { dot(qVecs[qi], memVecs[it]) }.take(3))
        }

        // ---- reranker
        say("Step 3: reranker")
        t0 = System.nanoTime()
        rc = b.loadAux(rerPath, LlamaBridge.KIND_RERANKER, threads)
        if (rc != 0) {
            say(loadError("reranker", rc, b))
            return
        }
        say("  loaded in " + secs(t0) + " s | " + b.auxInfo())

        t0 = System.nanoTime()
        var nScores = 0
        val rerankBest = IntArray(testCases.size)
        val rerankText = ArrayList<String>()
        for (qi in testCases.indices) {
            var best = -1
            var bestScore = -1e9f
            val parts = ArrayList<String>()
            for (mi in top[qi]) {
                val score = b.rerankScore(
                    testCases[qi].query.toByteArray(utf8),
                    testMemories[mi].toByteArray(utf8)
                )
                nScores++
                if (score <= -1e5f) {
                    say("  FAILED to score a memory. " + b.lastLog().takeLast(300))
                    b.unloadAux()
                    return
                }
                parts.add("" + (mi + 1) + " (" + "%.2f".format(score) + ")")
                if (score > bestScore) {
                    bestScore = score
                    best = mi
                }
            }
            rerankBest[qi] = best
            rerankText.add(parts.joinToString(", "))
        }
        val rerSec = (System.nanoTime() - t0) / 1e9
        say("  scored " + nScores + " pairs in " + "%.1f".format(rerSec) + " s")
        b.unloadAux()

        // ---- results
        say("")
        say("Memories:")
        for (i in testMemories.indices) say("  " + (i + 1) + ") " + testMemories[i])
        say("")

        var vectorOk = 0
        var rerankOk = 0
        for (qi in testCases.indices) {
            val c = testCases[qi]
            val vecTop = top[qi][0]
            val vecRight = vecTop == c.answer
            val rerRight = rerankBest[qi] == c.answer
            if (vecRight) vectorOk++
            if (rerRight) rerankOk++
            say("Question " + (qi + 1) + ": " + c.query)
            say("  right answer: memory " + (c.answer + 1))
            say("  vector search top 3: " + top[qi].joinToString(", ") { "" + (it + 1) } +
                (if (vecRight) "  -> OK" else "  -> WRONG"))
            say("  reranker scores: " + rerankText[qi] + "  -> best " + (rerankBest[qi] + 1) +
                (if (rerRight) "  OK" else "  WRONG"))
        }
        say("")
        say("Vector search alone: " + vectorOk + "/" + testCases.size)
        say("Vector search + reranker: " + rerankOk + "/" + testCases.size)
        say("Speed: embedding " + "%.0f".format(embSec * 1000 / nTexts) + " ms per text, reranking " +
            "%.0f".format(rerSec * 1000 / nScores) + " ms per pair")
        if (rerankOk == testCases.size) {
            say("RESULT: PASS. The backend works.")
        } else {
            say("RESULT: not perfect. Send this whole text to Claude.")
        }
    }

    private fun onNewChatClicked() {
        if (busy || !loaded) return
        val systemBytes = systemEdit.text.toString().toByteArray(Charsets.UTF_8)
        worker.execute { bridge?.resetChat(systemBytes) }
        outputText.text = ""
        statusText.text = "New chat started"
    }

    // ------------------------------------------------------------------ database self-test

    /** Tests the database with a separate test file; your real memory.db is only opened, never changed. */
    private fun onDbTestClicked() {
        if (busy) return
        if (!hasAllFilesAccess()) {
            toast("Tap 'Allow file access' first")
            return
        }
        setBusy(true)
        panelHolder.visibility = View.GONE
        updateToggleText()
        statusText.text = "Database self-test running..."
        appendOutput("\n\n")
        say("=== Database self-test ===")

        worker.execute {
            var ok = false
            try {
                val testPath = (File(HavenDb.DEFAULT_PATH).parent ?: "") + "/selftest.db"
                ok = DatabaseSelfTest { line -> say(line) }.run(HavenDb.DEFAULT_PATH, testPath)
            } catch (t: Throwable) {
                say("Database self-test crashed: " + t.javaClass.simpleName + " " + t.message)
            }
            runOnUiThread {
                statusText.text = if (ok) "Database self-test passed" else "Database self-test: problems found"
                setBusy(false)
            }
        }
    }

    // ------------------------------------------------------------------ search pipeline test

    /** Real models, real pipeline, a separate test database. Gemma is unloaded first (RAM). */
    private fun onSearchTestClicked() {
        if (busy) return
        if (!engineReady) {
            toast("The engine is not ready yet")
            return
        }
        if (!hasAllFilesAccess()) {
            toast("Tap 'Allow file access' first")
            return
        }
        val embPath = embedPathEdit.text.toString().trim()
        val rerPath = rerankPathEdit.text.toString().trim()
        if (!File(embPath).isFile) {
            statusText.text = "File not found: $embPath"
            return
        }
        if (!File(rerPath).isFile) {
            statusText.text = "File not found: $rerPath"
            return
        }
        val threads = threadsSpinner.selectedItemPosition + 1

        loaded = false          // Gemma is unloaded by the test
        setBusy(true)
        panelHolder.visibility = View.GONE
        updateToggleText()
        statusText.text = "Search pipeline test running..."
        appendOutput("\n\n")
        say("=== Search pipeline test ($threads threads) ===")

        worker.execute {
            val b = bridge
            var ok = false
            try {
                if (b == null) {
                    say("The engine is not available.")
                } else {
                    say("Unloading Gemma (RAM)...")
                    b.unload()
                    val testPath = (File(HavenDb.DEFAULT_PATH).parent ?: "") + "/searchtest.db"
                    ok = SearchPipelineTest(BridgeSearchModels(b)) { line -> say(line) }
                        .run(testPath, embPath, rerPath, threads)
                }
            } catch (e: SearchException) {
                say("SEARCH FAILED: " + e.message)
            } catch (t: Throwable) {
                say("Search test crashed: " + t.javaClass.simpleName + " " + t.message)
            }
            try { b?.unloadAux() } catch (t: Throwable) { }
            runOnUiThread {
                statusText.text = (if (ok) "Search pipeline test passed" else "Search pipeline test finished") +
                    ". Press Load model to chat again."
                setBusy(false)
            }
        }
    }

    private fun onSendClicked() {
        if (busy) return
        if (!loaded) {
            toast("Load the model first")
            return
        }
        val text = inputEdit.text.toString().trim()
        if (text.isEmpty()) return
        inputEdit.setText("")
        val thinking = thinkingBox.isChecked
        val split = newSplitter(showThinkBox.isChecked)
        splitter = split

        appendOutput("\n\nYou: $text\n\nGemma: ")
        setBusy(true)
        statusText.text = "Reading your message..."

        val listener = object : TokenListener {
            override fun onToken(bytes: ByteArray): Boolean {
                val piece = String(bytes, Charsets.UTF_8)
                runOnUiThread { split.feed(piece) }
                return true
            }
        }

        worker.execute {
            val b = bridge
            var stats = "ERROR: engine is not available"
            if (b != null) {
                stats = try {
                    b.generate(text.toByteArray(Charsets.UTF_8), 1024, thinking, listener)
                } catch (t: Throwable) {
                    "ERROR: " + t.javaClass.simpleName + " " + t.message
                }
            }
            val log = if (stats.startsWith("ERROR") && b != null) b.lastLog() else ""
            runOnUiThread {
                split.finish()
                if (split.inThinking) {
                    appendStyled("\n\n(No answer yet: it was stopped or ran out of tokens while thinking.)",
                        thinkingColor, 0.85f)
                }
                appendStyled("\n\n[" + stats + "]", thinkingColor, 0.8f)
                if (log.isNotEmpty()) appendStyled("\n" + log.takeLast(300), thinkingColor, 0.8f)
                statusText.text = "Ready"
                setBusy(false)
            }
        }
    }
}

/**
 * Splits Gemma's text stream into thinking and answer.
 * Gemma writes:  <|channel>thought ... thinking ... <channel|> answer
 * The marker words are never shown. Pure Kotlin (no Android), so it can be tested alone.
 */
class ThinkingSplitter(
    private val onThinkingStart: () -> Unit,
    private val onThinking: (String) -> Unit,
    private val onThinkingEnd: () -> Unit,
    private val onAnswer: (String) -> Unit
) {
    private val startMarker = "<|channel>"
    private val endMarker = "<channel|>"
    private val label = "thought"

    private var buf = StringBuilder()
    private var labelPending = false
    private var answerStarted = false

    /** True while the model is inside its thinking part. */
    var inThinking = false
        private set

    fun feed(piece: String) {
        buf.append(piece)
        drain(false)
    }

    /** Call once when the stream is over: sends out any text that was held back. */
    fun finish() {
        drain(true)
    }

    // How many characters at the end of text could be the beginning of marker.
    private fun partialLen(text: String, marker: String): Int {
        var k = minOf(text.length, marker.length - 1)
        while (k > 0) {
            if (text.endsWith(marker.substring(0, k))) return k
            k--
        }
        return 0
    }

    private fun emitAnswer(text: String) {
        var t = text
        if (!answerStarted) {
            t = t.trimStart()
            if (t.isEmpty()) return
            answerStarted = true
        }
        onAnswer(t)
    }

    private fun drain(finishing: Boolean) {
        while (true) {
            val text = buf.toString()

            if (!inThinking) {
                val i = text.indexOf(startMarker)
                if (i >= 0) {
                    if (i > 0) emitAnswer(text.substring(0, i))
                    buf = StringBuilder(text.substring(i + startMarker.length))
                    inThinking = true
                    labelPending = true
                    onThinkingStart()
                    continue
                }
                val hold = if (finishing) 0 else partialLen(text, startMarker)
                val n = text.length - hold
                if (n > 0) emitAnswer(text.substring(0, n))
                buf = StringBuilder(text.substring(n))
                return
            }

            if (labelPending) {
                // Right after the start marker Gemma writes the word "thought": hide it.
                if (!finishing && text.length <= label.length && label.startsWith(text)) return
                var rest = text
                if (rest.startsWith(label)) {
                    rest = rest.substring(label.length)
                    if (rest.startsWith("\n")) rest = rest.substring(1)
                }
                labelPending = false
                buf = StringBuilder(rest)
                continue
            }

            val j = text.indexOf(endMarker)
            if (j >= 0) {
                if (j > 0) onThinking(text.substring(0, j))
                buf = StringBuilder(text.substring(j + endMarker.length))
                inThinking = false
                answerStarted = false
                onThinkingEnd()
                continue
            }
            val hold = if (finishing) 0 else partialLen(text, endMarker)
            val n = text.length - hold
            if (n > 0) onThinking(text.substring(0, n))
            buf = StringBuilder(text.substring(n))
            return
        }
    }
}

/**
 * Database self-test. Runs on a background thread.
 * It uses a separate test file (selftest.db) that is deleted afterwards, so your real
 * memory.db is never changed. Vectors are small made-up ones (no model needed).
 */
class DatabaseSelfTest(private val say: (String) -> Unit) {

    private var passed = 0
    private var failed = 0

    private fun check(name: String, ok: Boolean, detail: String = "") {
        if (ok) {
            passed++
            say("  OK    $name")
        } else {
            failed++
            say("  FAIL  $name" + (if (detail.isNotEmpty()) "  ($detail)" else ""))
        }
    }

    private fun deleteFiles(path: String) {
        for (suffix in listOf("", "-journal", "-wal", "-shm")) File(path + suffix).delete()
    }

    private fun vec(vararg x: Float): FloatArray = x

    /** Returns true when every check passed. */
    fun run(realPath: String, testPath: String): Boolean {
        // ---- 1) the real database: only opened (this also creates it the first time)
        say("1) Opening your real database (nothing is changed)...")
        val real = HavenDb.open(realPath)
        say("   " + real.info())
        real.close()

        // ---- 2) a clean test database
        deleteFiles(testPath)
        var h = HavenDb.open(testPath)
        say("2) Test database created. Keyword index: " +
            (if (h.hasFts) "FTS5 (fast)" else "not available, slower word matching is used"))
        var mem = MemoryStore(h)
        var chats = ChatStore(h)

        // ---- 3) memories
        say("3) Memories")
        val ids = ArrayList<Long>()
        ids.add(mem.add("How did I fix the timing error in my counter project?",
            "Added a pull-down resistor on the clear pin of the 74HC161 counter IC.",
            listOf("74hc161", "counter", "clock skew")))
        ids.add(mem.add("Which tools do we use to build the phone app?",
            "Kotlin, llama.cpp and GitHub Actions. CPU only.", listOf("android", "kotlin", "github")))
        ids.add(mem.add("How fast is Gemma on my phone?",
            "About 7 tokens per second on the CPU of the Realme P3.", listOf("gemma", "speed", "realme")))
        ids.add(mem.add("Trip planning",
            "Booked train tickets to Jaipur for December.", listOf("travel", "jaipur")))
        ids.add(mem.add("C++ pointers",
            "The crash came from using a pointer after it was freed.", listOf("c++", "pointer", "segfault")))
        check("5 memories saved", mem.count() == 5 && ids.toSet().size == 5, "count " + mem.count())

        val again = mem.add("Trip planning", "Booked train tickets to Jaipur for December.", listOf("travel"))
        check("same memory is not saved twice", again == ids[3] && mem.count() == 5)

        val tagsBack = mem.getRows(listOf(ids[0])).firstOrNull()?.tags
        check("tags are stored tidy", tagsBack == "74hc161 counter clock skew", tagsBack ?: "null")

        check("keyword search finds the counter memory",
            mem.hitIds(listOf("counter resistor")) == listOf(ids[0]), mem.hitIds(listOf("counter resistor")).toString())
        check("keyword search works from the start of a word (point = pointer)",
            mem.hitIds(listOf("point")) == listOf(ids[4]), mem.hitIds(listOf("point")).toString())
        check("quick count is 0 when nothing matches", mem.quickCount(listOf("zebra giraffe")) == 0)
        check("filler words only: no search", mem.hitIds(listOf("the and")).isEmpty())

        // time filter: make the Jaipur memory 200 days old
        val old = LocalDate.now().minusDays(200).toString()
        h.db.execSQL("UPDATE memories SET event_date = ? WHERE id = ?", arrayOf<Any?>(old, ids[3]))
        check("time filter last_month hides an old memory", mem.hitIds(listOf("jaipur"), "last_month").isEmpty())
        check("time filter last_year and no filter find it",
            mem.hitIds(listOf("jaipur"), "last_year") == listOf(ids[3]) &&
                mem.hitIds(listOf("jaipur"), null) == listOf(ids[3]))

        // ---- 4) vectors (made-up 8-number vectors, one per memory)
        say("4) Vectors")
        check("all 5 memories wait for a vector", mem.pendingCount() == 5 && mem.pending().size == 5)
        val vectors = ids.indices.map { i -> FloatArray(8) { j -> if (j == i) 1f else 0f } }
        mem.setVecs(ids.indices.map { Pair(ids[it], vectors[it]) })
        check("vectors saved", mem.pendingCount() == 0)
        val roundTrip = bytesToFloats(floatsToBytes(vec(0.5f, -1.25f, 3.0e-5f)))
        check("vector bytes round trip", roundTrip.contentEquals(vec(0.5f, -1.25f, 3.0e-5f)))
        val query = vec(0f, 0.2f, 0f, 0.98f, 0f, 0f, 0f, 0f)
        val near = mem.nearest(query, null, 3)
        check("nearest vector is the right memory", near.firstOrNull() == ids[3] && near.size == 3, near.toString())
        check("a vector of the wrong size is skipped", mem.nearest(vec(1f, 0f, 0f), null, 3).isEmpty())

        // ---- 5) reading and deleting
        say("5) Reading and deleting")
        check("rows come back in the order asked",
            mem.getRows(listOf(ids[2], ids[0])).map { it.id } == listOf(ids[2], ids[0]))
        check("a missing id is skipped", mem.getRows(listOf(ids[0], 999999L)).size == 1)
        val recent = mem.recent(2)
        check("newest memory comes first", recent.size == 2 && recent[0].id == ids[4], recent.map { it.id }.toString())
        mem.delete(ids[1])
        check("delete removes the memory and its keyword entry",
            mem.count() == 4 && mem.hitIds(listOf("kotlin")).isEmpty())

        // ---- 6) chats
        say("6) Chats")
        val c1 = chats.createChat()
        check("new chat has the default title", chats.getChat(c1)?.title == ChatStore.DEFAULT_TITLE)
        chats.addMessage(c1, "user", "Can you tell me about my previous ai project")
        val expectedTitle = ChatStore.titleFrom("Can you tell me about my previous ai project")
        check("first message becomes the title",
            chats.getChat(c1)?.title == expectedTitle, chats.getChat(c1)?.title ?: "null")
        chats.addMessage(c1, "assistant", "You are building a private AI with long-term memory.", "Thinking: look in memory")
        chats.addMessage(c1, "user", "Thanks")
        check("title is set only once", chats.getChat(c1)?.title == expectedTitle)
        val msgs = chats.messages(c1)
        check("3 messages in order, thinking kept apart",
            msgs.map { it.role } == listOf("user", "assistant", "user") &&
                msgs[1].thinking == "Thinking: look in memory" && msgs[0].thinking == null)
        check("last message of a chat", chats.lastMessages(c1, 1).firstOrNull()?.content == "Thanks")

        val c2 = chats.createChat()
        chats.addMessage(c2, "user", "A second chat")
        check("most recent chat is first in the list", chats.listChats().firstOrNull()?.id == c2,
            chats.listChats().map { it.id }.toString())
        chats.renameChat(c1, "My AI chat")
        check("rename works", chats.getChat(c1)?.title == "My AI chat")
        chats.deleteChat(c2)
        check("delete removes the chat and its messages", chats.chatCount() == 1 && chats.messageCount(c2) == 0)

        // ---- 7) data stays on disk: close, open again
        say("7) Closing and opening the file again")
        h.close()
        h = HavenDb.open(testPath)
        mem = MemoryStore(h)
        chats = ChatStore(h)
        check("memories and vectors are still there",
            mem.count() == 4 && mem.nearest(query, null, 1).firstOrNull() == ids[3])
        check("chats and messages are still there", chats.chatCount() == 1 && chats.messageCount(c1) == 3)
        say("   " + h.info())

        h.close()
        deleteFiles(testPath)

        say("")
        say("RESULT: " + passed + " of " + (passed + failed) + " checks passed")
        say(if (failed == 0) "The database works." else "Some checks failed: send me this screen.")
        return failed == 0
    }
}

/**
 * Search pipeline test with the REAL embedder and reranker.
 * 23 made-up memories go into a separate test file (deleted afterwards). Six questions are
 * asked in different words; each should find its own memory. Times are shown for every step.
 */
class SearchPipelineTest(
    private val models: SearchModels,
    private val say: (String) -> Unit
) {

    private class Item(val question: String, val answer: String, val tags: List<String>)
    private class Case(val question: String, val expected: Int)

    private val items = listOf(
        // 0-5: the memories the questions should find
        Item("How did I fix the timing error in my counter project?",
            "Added a pull-down resistor on the clear pin of the 74HC161 counter IC. That fixed the clock skew.",
            listOf("74hc161", "counter", "clock skew", "resistor")),
        Item("Which tools do we use to build the phone app?",
            "Decided: Kotlin app, llama.cpp for the models, CPU only, and GitHub Actions to build the APK.",
            listOf("android", "kotlin", "llama.cpp", "github actions")),
        Item("How fast is Gemma on my phone?",
            "My phone is a Realme P3 with 8 GB RAM. Gemma 4 E2B runs at about 7 tokens per second on the CPU.",
            listOf("gemma", "speed", "realme p3", "tokens per second")),
        Item("Trip planning",
            "Planning a trip to Jaipur in December. Booked the train tickets and a small hotel near the old city.",
            listOf("travel", "jaipur", "train", "hotel")),
        Item("Why did my C++ program crash?",
            "Started learning pointers. The segmentation fault came from using a pointer after the memory was freed.",
            listOf("c++", "pointer", "segmentation fault")),
        Item("Which models find my old memories?",
            "Chose bge-m3 as the embedder and bge-reranker-v2-m3 as the reranker. Both passed the 5 out of 5 retrieval test.",
            listOf("bge-m3", "reranker", "retrieval", "embedder")),
        // 6-22: other notes that look a little similar (to make the search work)
        Item("Python practice", "Learning Python decorators and list comprehensions this week.", listOf("python", "learning")),
        Item("Electronics", "Bought a second-hand multimeter to check circuits on the breadboard.", listOf("multimeter", "breadboard")),
        Item("Phone battery", "The battery of my phone drains fast when the screen stays on.", listOf("battery", "phone")),
        Item("Cooking", "Recipe for dal tadka: boil lentils, then fry cumin, garlic and chilli.", listOf("recipe", "dal")),
        Item("Reading", "Reading a book about operating systems, chapter on virtual memory and paging.", listOf("book", "operating systems")),
        Item("Goa photos", "The beach holiday photos from Goa need to be sorted and backed up.", listOf("photos", "goa")),
        Item("Termux git", "Set up git on Termux and pushed the first commit to the repository.", listOf("termux", "git")),
        Item("Gemma output", "Gemma writes thinking text before the answer; the screen shows it in grey.", listOf("gemma", "thinking")),
        Item("Math homework", "Derivative of sin x times x squared, then the integral of x times e to the x.", listOf("math", "homework")),
        Item("Daily plan", "Wake up at six, walk thirty minutes, study C++ for two hours.", listOf("plan", "study")),
        Item("Shopping", "Need to buy a USB-C cable and a power bank.", listOf("shopping", "cable")),
        Item("Memory database", "The SQLite database stores raw questions and answers with tags.", listOf("sqlite", "database")),
        Item("Space", "Watched a documentary about space telescopes and dark matter.", listOf("space", "documentary")),
        Item("Monitor", "The monitor flickers when the HDMI cable is loose.", listOf("monitor", "hdmi")),
        Item("Chess", "Practising chess openings: the Italian game and the Queen's gambit.", listOf("chess")),
        Item("Plants", "Water the plants twice a week and clean the balcony.", listOf("plants", "home")),
        Item("Family call", "Called mom on Sunday; she asked about the exam results.", listOf("family", "call"))
    )

    private val cases = listOf(
        Case("How did I solve the timing problem in my digital logic counter?", 0),
        Case("Which tools did we pick for building the phone app?", 1),
        Case("How fast is the language model on my handset?", 2),
        Case("What are my travel arrangements for winter?", 3),
        Case("Why did my program crash when I used a dangling pointer?", 4),
        Case("Which neural networks do I use for searching past notes?", 5)
    )

    private fun secs(t0: Long): String = "%.1f".format((System.nanoTime() - t0) / 1e9)

    private fun deleteFiles(path: String) {
        for (suffix in listOf("", "-journal", "-wal", "-shm")) File(path + suffix).delete()
    }

    /** Returns true when every question finds its own memory in the best 3. */
    fun run(dbPath: String, embPath: String, rerPath: String, threads: Int): Boolean {
        deleteFiles(dbPath)
        val h = HavenDb.open(dbPath)
        try {
            val store = MemoryStore(h)
            val ids = items.map { store.add(it.question, it.answer, it.tags) }
            say("Saved " + ids.size + " test memories in a separate file.")
            say("Candidates per search: " + Retrieval.DEFAULT_VECTOR_CANDIDATES + " by vector + " +
                Retrieval.DEFAULT_KEYWORD_CANDIDATES + " by keyword.")
            say("")

            val retrieval = Retrieval(models, store, embPath, rerPath, threads) { say("   " + it) }

            say("1) Embedding all memories (bge-m3)...")
            var t0 = System.nanoTime()
            val n = retrieval.embedPending()
            say("   " + n + " memories in " + secs(t0) + " s")
            say("")

            say("2) Searching (embedder + reranker are loaded for every search)...")
            var top1 = 0
            var top3 = 0
            val searchStart = System.nanoTime()
            for ((i, c) in cases.withIndex()) {
                say("")
                say("Q" + (i + 1) + ": " + c.question)
                t0 = System.nanoTime()
                val hits = retrieval.search(c.question)
                val took = secs(t0)
                val expectedId = ids[c.expected]
                for ((rank, hit) in hits.withIndex()) {
                    say("   " + (rank + 1) + ". score " + "%.2f".format(hit.score) + "  " +
                        hit.question.take(45) + (if (hit.id == expectedId) "   <-- right" else ""))
                }
                val first = hits.firstOrNull()?.id == expectedId
                val inThree = hits.any { it.id == expectedId }
                if (first) top1++
                if (inThree) top3++
                say("   " + (if (first) "OK (first)" else if (inThree) "OK (in the best 3, not first)" else "WRONG") +
                    " | search time " + took + " s")
            }
            say("")
            say("RESULT: right memory first " + top1 + "/" + cases.size +
                " | in the best 3: " + top3 + "/" + cases.size +
                " | average search time " + "%.1f".format((System.nanoTime() - searchStart) / 1e9 / cases.size) + " s")
            say(if (top3 == cases.size) "The search pipeline works." else "Some answers are wrong: send me this screen.")
            return top3 == cases.size
        } finally {
            h.close()
            deleteFiles(dbPath)
        }
    }
}
