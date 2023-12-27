import streamlit as st
from streamlit.components.v1 import html

# 嵌入 HTML 和 JavaScript 代码
audio_component = html("""
<html>
<body>
<button id="start">开始录音</button>
<button id="stop">停止录音</button>
<audio id="audio" controls></audio>
<script>
  let mediaRecorder;
  let audioChunks = [];
  let audio = document.getElementById("audio");

  document.getElementById("start").addEventListener("click", function() {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        audioChunks = [];

        mediaRecorder.addEventListener("dataavailable", event => {
          audioChunks.push(event.data);
        });

        mediaRecorder.addEventListener("stop", () => {
          const audioBlob = new Blob(audioChunks);
          const audioUrl = URL.createObjectURL(audioBlob);
          audio.src = audioUrl;
          // 这里可以将 audioBlob 发送到 Streamlit 服务器进行处理
        });
      });
  });

  document.getElementById("stop").addEventListener("click", function() {
    mediaRecorder.stop();
  });
</script>
</body>
</html>
""", height=200)

# Streamlit 代码
if st.button('处理音频'):
    # 这里可以添加处理音频的代码
    st.write("处理音频...")