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

// Constants and Configurations
#define I2S_NUM           I2S_NUM_0
#define I2S_SAMPLE_RATE   16000
#define I2S_SAMPLE_BITS   32
#define I2S_READ_LEN      (I2S_SAMPLE_RATE * I2S_SAMPLE_BITS / 8 / 1000)
#define BUFFER_SIZE       (1024)
#define RX_PIN 16  // Define your RX pin number for Serial2
#define TX_PIN 17  // Define your TX pin number for Serial2


//const char* ssid = "SANTYRJ_25 7696";
//const char* password = "Burma007";
const char* ssid = "TrueFlame's iPhone";
const char* password = "sam123456";
const char* apiKey = "f120f78ce9814f66acf59f062d431c55";
const char* serverName = "https://api.assemblyai.com/v2/transcript";

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

// Function Declarations
void initializeI2S();
void initializeTensorFlow();
void connectToWiFi();
void recordAudioToSPIFFS();
void sendAudioFromSPIFFS();
void RespondToCommand(const char* command, float score, tflite::ErrorReporter* error_reporter);

// Function Definitions
void initializeI2S() {
    Serial.println("Initializing I2S...");
    i2s_driver_install(I2S_NUM, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM, &pin_config);
    i2s_set_clk(I2S_NUM, I2S_SAMPLE_RATE, static_cast<i2s_bits_per_sample_t>(I2S_SAMPLE_BITS), I2S_CHANNEL_MONO);
    Serial.println("I2S initialized.");
}

void initializeTensorFlow() {
    Serial.println("Initializing TensorFlow...");
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
    Serial.println("TensorFlow initialized.");
}

void connectToWiFi() {
    Serial.println("Connecting to WiFi...");
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.print(".");
    }
    Serial.println("\nConnected to WiFi!");
}

void recordAudioToSPIFFS() {
    Serial.println("Recording audio to SPIFFS...");
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
    Serial.println("Sending audio from SPIFFS...");
    File audioFile = SPIFFS.open(audioFilePath, "r");
    if (!audioFile) {
        Serial.println("Failed to open file for reading");
        return;
    }

    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverName);
        http.addHeader("authorization", apiKey);
        http.addHeader("Content-Type", "audio/raw");
        
        int httpResponseCode = http.POST((uint8_t *)i2s_read_buff, audioFile.size());
        if (httpResponseCode > 0) {
            String response = http.getString();
            Serial2.println(response);  // Send the response to the second ESP32
            Serial.println("Audio sent successfully and response received.");
        } else {
            Serial.println("Error in sending POST request");
        }
        audioFile.close();
    } else {
        Serial.println("Not connected to WiFi. Cannot send audio.");
    }
}

void RespondToCommand(const char* command, float score, tflite::ErrorReporter* error_reporter) {
    Serial.printf("Command recognized: %s with score: %f\n", command, score);
    if (strcmp(command, "YES") == 0) {
        digitalWrite(2, HIGH);
        Serial.println("LED turned ON.");
        recordAudioToSPIFFS();
        digitalWrite(2, LOW);
        Serial.println("LED turned OFF.");
        sendAudioFromSPIFFS();
    } else if (strcmp(command, "NO") == 0) {
        Serial.println("NO command detected.");
    }
}



void setup() {
    Serial.begin(115200);
    Serial.println("Starting setup...");
    pinMode(2, OUTPUT);
    initializeI2S();
    initializeTensorFlow();
    if (!SPIFFS.begin(true)) {
        Serial.println("An error has occurred while mounting SPIFFS");
        return;
    }
    connectToWiFi();
    Serial2.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN); // Initialize Serial2
    Serial.println("Setup completed.");
}

void loop() {
    Serial.println("Starting loop iteration...");
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
    Serial.println("Loop iteration completed.");
}
