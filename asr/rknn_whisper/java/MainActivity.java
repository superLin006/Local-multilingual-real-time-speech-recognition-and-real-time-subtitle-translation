public class MainActivity extends AppCompatActivity {
    private WhisperRecognizer recognizer;
    private TextView resultView;
    private Button toggleButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        resultView = findViewById(R.id.result_text);
        toggleButton = findViewById(R.id.toggle_button);

        recognizer = new WhisperRecognizer();
        
        // 初始化
        String modelPath = "/sdcard/whisper_models/";
        if (!recognizer.initialize(modelPath, "en")) {
            Toast.makeText(this, "初始化失败", Toast.LENGTH_SHORT).show();
            return;
        }

        toggleButton.setOnClickListener(v -> {
            if (!recognizer.isRecording) {
                // 开始识别
                resultView.setText("");
                recognizer.startRecognition(new WhisperRecognizer.RecognitionCallback() {
                    @Override
                    public void onNewText(String text) {
                        // ✅ 在UI线程更新界面
                        runOnUiThread(() -> {
                            resultView.append(text);
                        });
                    }

                    @Override
                    public void onError(String error) {
                        runOnUiThread(() -> {
                            Toast.makeText(MainActivity.this, 
                                         "错误: " + error, 
                                         Toast.LENGTH_SHORT).show();
                        });
                    }

                    @Override
                    public void onComplete() {
                        runOnUiThread(() -> {
                            String stats = recognizer.getStatistics();
                            Toast.makeText(MainActivity.this, 
                                         "完成: " + stats, 
                                         Toast.LENGTH_LONG).show();
                        });
                    }
                });
                toggleButton.setText("停止");
            } else {
                // 停止识别
                recognizer.stopRecognition();
                toggleButton.setText("开始");
                
                // 显示完整结果
                String full = recognizer.getFullResult();
                resultView.setText("完整结果:\n" + full);
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        recognizer.release();
    }
}

/*  
 ## 3. 性能优化建议

- **推荐使用阻塞版本**（`getResult(100)`）：更省CPU，等待有结果才唤醒
- **非阻塞版本**（`tryGetResult()`）：适合需要高频率轮询的场景，但需要手动 sleep

## 4. 完整流程
```
用户点击开始
    ↓
initialize() - 初始化模型（启动后台推理线程）
    ↓
startRecognition() - 启动2个线程：
    ├─ 录音线程：麦克风 → feedAudioAsync() → 音频缓冲区
    └─ 结果线程：tryGetResult() → callback.onNewText()
    ↓
用户点击停止
    ↓
stopRecognition() - 依次：
    ├─ 停止录音
    ├─ finishInput() - 标记输入结束
    ├─ waitForCompletion() - 等待剩余推理
    ├─ 停止结果线程
    └─ callback.onComplete()
    ↓
release() - 释放所有资源
*/

