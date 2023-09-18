#include <TensorFlowLite_ESP32.h>
#include "main_functions.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <driver/i2s.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <FS.h>
#include <SPIFFS.h>

extern "C" {
    #include "main.h"
}

// Constants and Configuration
#define BUFFER_SIZE       (1024)
#define I2S_NUM           I2S_NUM_0
#define I2S_SAMPLE_RATE   16000
#define I2S_SAMPLE_BITS   32  // Change from 16 to 32 bits
#define I2S_READ_LEN      (I2S_SAMPLE_RATE * I2S_SAMPLE_BITS / 8 / 1000) // 1ms worth of samples

const char* ssid = "SANTYRJ_25 7696";
const char* password = "Burma007";
const char* gcpServerName = "https://speech.googleapis.com/v1/speech:recognize";
const char* gcpApiKey = "YOUR_GOOGLE_CLOUD_API_KEY";

const i2s_config_t i2s_config = {
    .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate =  I2S_SAMPLE_RATE,
    .bits_per_sample = static_cast<i2s_bits_per_sample_t>(I2S_SAMPLE_BITS),
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = static_cast<i2s_comm_format_t>(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 2,
    .dma_buf_len = 1024,
    .use_apll = false
};

const i2s_pin_config_t pin_config = {
    .bck_io_num = GPIO_NUM_32,
    .ws_io_num = GPIO_NUM_25,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = GPIO_NUM_26
};

// Global Variables
namespace {
    uint8_t i2s_read_buff[BUFFER_SIZE];
    const char* audioFilePath = "/audio.raw";
}

// TensorFlow Lite related global variables
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;

// Function Definitions
void initializeI2S() {
    i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM, &pin_config);
    i2s_set_clk(I2S_NUM, I2S_SAMPLE_RATE, static_cast<i2s_bits_per_sample_t>(I2S_SAMPLE_BITS), I2S_CHANNEL_MONO);
}

void initializeTensorFlow() {
    tflite::InitializeTarget();
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    model = tflite::GetModel(g_model);

    static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddReshape();

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    model_input = interpreter->input(0);
    model_input_buffer = model_input->data.int8;

    static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer);
    feature_provider = &static_feature_provider;

    static RecognizeCommands static_recognizer(error_reporter);
    recognizer = &static_recognizer;
}

void connectToWiFi() {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi!");
}

void recordAudioToSPIFFS() {
    Serial.println("Recording audio...");
    File audioFile = SPIFFS.open(audioFilePath, "w");
    if (!audioFile) {
        Serial.println("Failed to create file");
        return;
    }

    for (int i = 0; i < 10 * (I2S_SAMPLE_RATE * 4 / BUFFER_SIZE); i++) {
        size_t bytes_read;
        i2s_read(I2S_NUM, i2s_read_buff, BUFFER_SIZE, &bytes_read, portMAX_DELAY);
        audioFile.write((byte*)i2s_read_buff, bytes_read);
    }

    audioFile.close();
    Serial.println("Audio recording completed.");
}

void sendAudioFromSPIFFS() {
    Serial.println("Sending audio to Google Cloud Platform...");
    File audioFile = SPIFFS.open(audioFilePath, "r");
    if (!audioFile) {
        Serial.println("Failed to open file for reading");
        return;
    }

    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(gcpServerName);
        http.addHeader("Authorization", String("Bearer ") + gcpApiKey);
        http.addHeader("Content-Type", "application/json");

        String jsonPayload = R"({
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": "en-US"
            },
            "audio": {
                "content": ")" + audioFile.readString() + R"("
            }
        })";

        int httpResponseCode = http.POST(jsonPayload);
        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial.println("Received response:");
            Serial.println(response);
        } else {
            Serial.print("Error sending HTTP POST request. Error code: ");
            Serial.println(httpResponseCode);
        }

        audioFile.close();
    } else {
        Serial.println("Not connected to WiFi.");
    }
}

void RespondToCommand(const char* command, float score, tflite::ErrorReporter* error_reporter) {
    if (strcmp(command, "YES") == 0) {
        recordAudioToSPIFFS();
        sendAudioFromSPIFFS();
    } else if (strcmp(command, "NO") == 0) {
        // Do something for "NO"
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(2, OUTPUT);  // Set GPIO 2 as output for the built-in LED

    initializeI2S();
    initializeTensorFlow();
    connectToWiFi();
}

void loop() {
    const int32_t current_time = LatestAudioTimestamp();
    int how_many_new_slices = 0;
    feature_provider->PopulateFeatureData(error_reporter, previous_time, current_time, &how_many_new_slices);
    previous_time = current_time;

    if (how_many_new_slices > 0) {
        for (int i = 0; i < kFeatureElementCount; i++) {
            model_input_buffer[i] = feature_buffer[i];
        }

        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed");
            return;
        }

        const float threshold = 0.5f;
        const int32_t minimum_count = 3;
        TfLiteTensor* output = interpreter->output(0);

        const char* found_command;
        uint8_t score;
        bool is_new_command;
        recognizer->ProcessLatestResults(output, current_time, &found_command, &score, &is_new_command);

        if (found_command[0] == 'Y' || found_command[0] == 'N') {
            RespondToCommand(found_command, score, error_reporter);
        }
    }

    int32_t i2s_read_buff[320];
    size_t bytes_read;
    i2s_read(I2S_NUM, (char *)i2s_read_buff, I2S_READ_LEN, &bytes_read, portMAX_DELAY);
}
