package com.haven.chat

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.text.InputType
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
    private lateinit var systemEdit: EditText
    private lateinit var loadBtn: Button
    private lateinit var newChatBtn: Button
    private lateinit var infoText: TextView
    private lateinit var scroll: ScrollView
    private lateinit var outputText: TextView
    private lateinit var inputEdit: EditText
    private lateinit var sendBtn: Button
    private lateinit var stopBtn: Button

    private val ctxChoices = listOf(2048, 4096, 8192)

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
        thinkingBox.text = "Thinking on (shows raw thinking text)"
        thinkingBox.textSize = 13f
        panel.addView(thinkingBox)

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

        infoText = TextView(this)
        infoText.textSize = 11f
        panel.addView(infoText)

        root.addView(panel)

        // Output
        scroll = ScrollView(this)
        outputText = TextView(this)
        outputText.textSize = 15f
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
            panel.visibility = if (panel.visibility == View.VISIBLE) View.GONE else View.VISIBLE
            updateToggleText()
        }
        permBtn.setOnClickListener { askAllFilesAccess() }
        loadBtn.setOnClickListener { onLoadClicked() }
        newChatBtn.setOnClickListener { onNewChatClicked() }
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
        toggleBtn.text = if (panel.visibility == View.VISIBLE) "Hide settings" else "Settings"
    }

    private fun setBusy(b: Boolean) {
        busy = b
        loadBtn.isEnabled = !b
        newChatBtn.isEnabled = !b
        sendBtn.isEnabled = !b
        stopBtn.isEnabled = b
    }

    private fun appendOutput(s: String) {
        outputText.append(s)
        scroll.post { scroll.scrollTo(0, outputText.height) }
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
                    panel.visibility = View.GONE
                    updateToggleText()
                } else {
                    statusText.text = "Load failed (code $rc). " + log.takeLast(500)
                }
                setBusy(false)
            }
        }
    }

    private fun onNewChatClicked() {
        if (busy || !loaded) return
        val systemBytes = systemEdit.text.toString().toByteArray(Charsets.UTF_8)
        worker.execute { bridge?.resetChat(systemBytes) }
        outputText.text = ""
        statusText.text = "New chat started"
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

        appendOutput("\n\nYou: $text\n\nGemma: ")
        setBusy(true)
        statusText.text = "Reading your message..."

        val listener = object : TokenListener {
            override fun onToken(bytes: ByteArray): Boolean {
                val piece = String(bytes, Charsets.UTF_8)
                runOnUiThread { appendOutput(piece) }
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
                appendOutput("\n\n[" + stats + "]")
                if (log.isNotEmpty()) appendOutput("\n" + log.takeLast(300))
                statusText.text = "Ready"
                setBusy(false)
            }
        }
    }
}
