plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.haven.chat"
    compileSdk = 34

    // Installed by the GitHub workflow (sdkmanager) before the build.
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.haven.chat"
        minSdk = 26
        targetSdk = 34
        versionCode = 2
        versionName = "0.2"

        // Phone is arm64-v8a only (no other ABI is built)
        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                // Release is essential: a Debug build of llama.cpp is extremely slow.
                arguments += "-DCMAKE_BUILD_TYPE=Release"
                arguments += "-DANDROID_STL=c++_shared"

                // llama.cpp: shared libraries, CPU only
                arguments += "-DBUILD_SHARED_LIBS=ON"
                arguments += "-DLLAMA_BUILD_APP=OFF"
                arguments += "-DLLAMA_BUILD_COMMON=ON"
                arguments += "-DLLAMA_OPENSSL=OFF"

                // ggml: build several CPU variants and pick the best one at run time
                arguments += "-DGGML_NATIVE=OFF"
                arguments += "-DGGML_BACKEND_DL=ON"
                arguments += "-DGGML_CPU_ALL_VARIANTS=ON"
                arguments += "-DGGML_LLAMAFILE=OFF"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }

    // The CPU variant libraries are loaded from the app's native lib folder,
    // so they must be extracted to disk when the app is installed.
    packaging {
        jniLibs {
            useLegacyPackaging = true
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
