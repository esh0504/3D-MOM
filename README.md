# Speech2Motion Demo

`demo.py`로 `.blend`를 생성하고, 같은 실행에서 MP4 렌더링까지 진행하는 데모입니다.

## 1) 폴더 구조

아래 경로를 기준으로 실행합니다.

```text
speech2motion/
  demo.py
  sample.wav
  vertex_groups_v3.json
  assets/
    mouth.blend
    mouth.jpg
    Mouth_DIFFUSE_02.png
  charsiu_ko/
    charsiu_predictive_aligner_ko.py
    k_lipmotion.json
    k_tonguemotion.json
    k_vocal2model.json
    k_vocab-ctc.json
  modules/
    blend_render.py
    comfort_viz.py
    GT_Eval.py
    module.py
  weights/
    model_ver_multi_w2v2_fc_10ms_ptk_weighted_v1/
      model.safetensors                              # 외부 다운로드
      config.json
      vocab.json
      processor/
  tools/
    blender/
```

## 2) Python 환경

아래처럼 conda 환경을 만든 뒤 `requirements.txt`를 설치하면 됩니다.

```cmd
conda create -y -n XAI_demo python=3.11
conda run -n XAI_demo python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r C:\path\to\speech2motion\requirements.txt
```

`requirements.txt`에는 `bpy==4.2.13`, `numpy==1.26.4`, `torch==2.3.1+cu121`가 포함되어 있습니다.

## 3) 가중치(weight) 준비

가중치는 아래 Google Drive에서 다운로드한 뒤,
`weights/model_ver_multi_w2v2_fc_10ms_ptk_weighted_v1/model.safetensors` 경로에 두면 됩니다.

- https://drive.google.com/file/d/1pbrb8Y1MOblaUSuI9CaZ6LN4cCpK4z-5/view?usp=drive_link

즉 최종 경로는 아래와 같습니다.

```text
./weights/model_ver_multi_w2v2_fc_10ms_ptk_weighted_v1/model.safetensors
```

## 4) Blender 준비

Blender는 저장소에 포함하지 않습니다.
Linux든 Windows든 Blender 4.2.9를 내려받아 압축을 푼 뒤, 실행 파일 경로를 `BLENDER_BIN`으로 직접 지정해서 실행하면 됩니다.

1. Linux에서 준비:
   - https://download.blender.org/release/Blender4.2/blender-4.2.9-linux-x64.tar.xz

```bash
cd /path/to/speech2motion
mkdir -p ./tools/blender
wget -O ./tools/blender/blender-4.2.9-linux-x64.tar.xz https://download.blender.org/release/Blender4.2/blender-4.2.9-linux-x64.tar.xz
tar -xJf ./tools/blender/blender-4.2.9-linux-x64.tar.xz -C ./tools/blender
```

2. Windows(cmd)에서 준비:
   - https://download.blender.org/release/Blender4.2/blender-4.2.9-windows-x64.zip

```cmd
cd C:\path\to\speech2motion
mkdir tools\blender
curl.exe -L -o tools\blender\blender-4.2.9-windows-x64.zip https://download.blender.org/release/Blender4.2/blender-4.2.9-windows-x64.zip
tar -xf tools\blender\blender-4.2.9-windows-x64.zip -C tools\blender
dir tools\blender\blender-4.2.9-windows-x64\blender.exe
```

3. 실행 파일 경로 예시:

```text
Linux:   ./tools/blender/blender-4.2.9-linux-x64/blender
Windows: .\tools\blender\blender-4.2.9-windows-x64\blender.exe
```

## 5) 실행

저장소에는 `sample.wav`가 포함되어 있으며, 이 파일은 `16 kHz` 예제 입력 파일입니다.
별도 WAV가 없으면 먼저 `./sample.wav`로 실행해보면 됩니다.

```bash
cd /path/to/speech2motion
export BLENDER_BIN=./tools/blender/blender-4.2.9-linux-x64/blender
python3 ./demo.py \
  --input "./sample.wav" \
  --model_path ./weights/model_ver_multi_w2v2_fc_10ms_ptk_weighted_v1 \
  --setting_dir ./charsiu_ko \
  --save_path ./results \
  --base_modelpath ./assets/mouth.blend \
  --mode prob \
  --n_prob 5 \
  --log True \
  --dir True \
  --blend_tag "" \
  --render True \
  --render_blender_bin "$BLENDER_BIN" \
  --comfort_viz True \
  --comfort_viz_json ./vertex_groups_v3.json \
  --comfort_alpha_front 0.7 \
  --comfort_alpha_side 0.4
```

Windows에서는 `cmd.exe` 기준으로 아래처럼 실행하면 됩니다.

```cmd
cd C:\path\to\speech2motion
set BLENDER_BIN=.\tools\blender\blender-4.2.9-windows-x64\blender.exe
python .\demo.py --input ".\sample.wav" --model_path .\weights\model_ver_multi_w2v2_fc_10ms_ptk_weighted_v1 --setting_dir .\charsiu_ko --save_path .\results --base_modelpath .\assets\mouth.blend --mode prob --n_prob 5 --log True --dir True --blend_tag "" --render True --render_blender_bin "%BLENDER_BIN%" --comfort_viz True --comfort_viz_json .\vertex_groups_v3.json --comfort_alpha_front 0.7 --comfort_alpha_side 0.4
```

## 6) 주요 옵션

- 입력 wav: `--input`
- 저장 경로: `--save_path`
- 모델 경로: `--model_path ./weights/model_ver_multi_w2v2_fc_10ms_ptk_weighted_v1`
- Blender 경로(Linux 예시): `./tools/blender/blender-4.2.9-linux-x64/blender`
- Blender 경로(Windows 예시): `.\tools\blender\blender-4.2.9-windows-x64\blender.exe`

## 7) 출력

기본 출력 경로 예시:

```text
./results/<wav_stem>/prob_5/<wav_stem>.blend
./results/<wav_stem>/prob_5/<wav_stem>.mp4
```

## 8) GPU 렌더링 (IMPORTANT)

렌더링은 반드시 GPU가 연결된 상태에서 수행하는 것을 권장합니다.
GPU가 연결되지 않으면 소프트웨어 렌더링으로 fallback되어 처리 시간이 크게 증가할 수 있습니다.

예시 기준(480 frame 렌더):

- GPU 미연결(CPU로 동작): 약 `1시간`
- GPU 연결 (RTX 3090): 대략 `약 1분 40초`
- GPU 연결 (GTX 1080 Ti): 약 `4분 10초`

실제 소요 시간은 장면 복잡도, Blender 설정, 드라이버/컨테이너 환경에 따라 달라질 수 있습니다.
