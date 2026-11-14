package com.haven.chat

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.widget.TextView

/** Phase 1: a "hello" screen that proves the GitHub build pipeline works. */
class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val view = TextView(this).apply {
            textSize = 20f
            setPadding(48, 120, 48, 48)
            text = "Haven Chat\n\nHello! The build pipeline works.\n\n" +
                "Phone ABI: ${Build.SUPPORTED_ABIS.joinToString()}"
        }
        setContentView(view)
    }
}
