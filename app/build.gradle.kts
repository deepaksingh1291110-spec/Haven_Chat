plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.haven.chat"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.haven.chat"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"

        // Phone is arm64-v8a only (no other ABI is built)
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}
