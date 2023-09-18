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
#include <ArduinoJson.h>
#include "Audio.h"

extern "C" {
    #include "main.h"
}


const char* chatgpt_token = "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";
const char* temperature = "0";
const char* max_tokens = "45";
String Question = "";
#define I2S_DOUT      25
#define I2S_BCLK      27
#define I2S_LRC       26
String transcribedText = "Placeholder text"; 
#define I2S_NUM           I2S_NUM_0
#define I2S_SAMPLE_RATE   16000
#define I2S_SAMPLE_BITS   32
#define I2S_READ_LEN      (I2S_SAMPLE_RATE * I2S_SAMPLE_BITS / 8 / 1000)

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


const char* ssid = "your-SSID";
const char* password = "your-PASSWORD";
const char* apiKey = "YOUR_ASSEMBLY_AI_KEY";
const char* serverName = "https://api.assemblyai.com/v2/transcript";
#define BUFFER_SIZE     (1024)
#define I2S_BCK_IO      (GPIO_NUM_33)
#define I2S_WS_IO       (GPIO_NUM_25)
#define I2S_DO_IO       (GPIO_NUM_26)
#define I2S_DI_IO       (GPIO_NUM_27)


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
Audio audio;

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
}

void sendAudioFromSPIFFS() {
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
        
        while (audioFile.available()) {
            int bytesRead = audioFile.read((byte*)i2s_read_buff, BUFFER_SIZE);
            http.POST((uint8_t *)i2s_read_buff, bytesRead);
        }

        audioFile.close();
    }
}

String getOpenAIResponse(String question) {
    HTTPClient https;
    if (https.begin("https://api.openai.com/v1/completions")) {
        https.addHeader("Content-Type", "application/json");
        String token_key = String("Bearer ") + chatgpt_token;
        https.addHeader("Authorization", token_key);
        String payload = "{\"model\": \"text-davinci-003\", \"prompt\": " + question + ", \"temperature\": " + temperature + ", \"max_tokens\": " + max_tokens + "}";
        int httpCode = https.POST(payload);
        if (httpCode == HTTP_CODE_OK || httpCode == HTTP_CODE_MOVED_PERMANENTLY) {
            String responsePayload = https.getString();
            DynamicJsonDocument doc(1024);
            deserializeJson(doc, responsePayload);
            String answer = doc["choices"][0]["text"];
            return answer;
        }
        https.end();
    }
    return "";
}

void RespondToCommand(const char* command, float score, tflite::ErrorReporter* error_reporter) {
    if (strcmp(command, "YES") == 0) {
        digitalWrite(2, HIGH);
        recordAudioToSPIFFS();
        digitalWrite(2, LOW);
        sendAudioFromSPIFFS();
        String response = getOpenAIResponse(transcribedText);
        audio.connecttospeech(response.c_str(), "en");
    } else if (strcmp(command, "NO") == 0) {
        // Do something for "NO"
    }
}

void setup() {
    pinMode(2, OUTPUT);
    initializeI2S();
    initializeTensorFlow();
    Serial.begin(115200);
    if (!SPIFFS.begin(true)) {
        Serial.println("An error has occurred while mounting SPIFFS");
        return;
    }
    connectToWiFi();
    audio.setPinout(I2S_BCK_IO, I2S_WS_IO, I2S_DO_IO);
    audio.setVolume(100);
}

void loop() {
    int32_t i2s_read_buff[320];
    size_t bytes_read;

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

    i2s_read(I2S_NUM, (char *)i2s_read_buff, I2S_READ_LEN, &bytes_read, portMAX_DELAY);
    
    Serial.print("Ask your Question : ");
    while (!Serial.available()) {
        audio.loop();
    }
    while (Serial.available()) {
        char add = Serial.read();
        Question = Question + add;
        delay(1);
    }
    int len = Question.length();
    Question = Question.substring(0, (len - 1));
    Question = "\"" + Question + "\"";
    Serial.println(Question);

    String response = getOpenAIResponse(Question);
    audio.connecttospeech(response.c_str(), "en");
    Question = "";
}
