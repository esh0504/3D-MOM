# /root/workspace/charsiu_ko/charsiu_predictive_aligner_ko.py

import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import json

# class charsiu_predictive_aligner_ko:
#     def __init__(self, model_path, processor_path, vocab_path, device=None):
#         # 1. processor 먼저 정의
#         self.processor = Wav2Vec2Processor.from_pretrained(processor_path)

#         # 2. model 정의
#         self.model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()

#         # 3. blank_id 정의 ← 이제 self.processor가 있으니까 여기서 가능
#         self.blank_id = self.processor.tokenizer.pad_token_id

#         # 4. vocab 로드
#         with open(vocab_path, "r", encoding="utf-8") as f:
#             self.id2label = {v: k for k, v in json.load(f).items()}

#         # 5. device 설정 및 모델 이동
#         self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)

#     def align_segmented(self, audio, min_duration=3, merge_repeats=True):
#         """
#         charsiu-style CTC decoding + duration smoothing/merging 적용 버전
#         """
#         if isinstance(audio, str):
#             audio, sr = sf.read(audio)
#             if sr != 16000:
#                 raise ValueError(f"Expected 16kHz sampling rate, got {sr}Hz")

#         inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         with torch.no_grad():
#             logits = self.model(**inputs).logits  # (1, T, V)
#             probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (T, V)

#         pred_ids = np.argmax(probs, axis=-1)  # (T,)

#         segments = []
#         prev = None
#         start = 0
#         for i, p in enumerate(pred_ids):
#             if p == self.blank_id:
#                 continue

#             # phoneme 변화 발생
#             if p != prev:
#                 duration = i - start
#                 if prev is not None and duration >= min_duration:
#                     if merge_repeats and segments and prev == segments[-1][2]:
#                         # ✅ 같은 phoneme이면 이전 구간에 merge
#                         segments[-1] = (segments[-1][0], i * 0.01, prev)
#                     else:
#                         segments.append((start * 0.01, i * 0.01, prev))
#                 start = i
#                 prev = p

#         # 마지막 segment 처리
#         if prev is not None and (len(pred_ids) - start) >= min_duration:
#             if merge_repeats and segments and prev == segments[-1][2]:
#                 segments[-1] = (segments[-1][0], len(pred_ids) * 0.01, prev)
#             else:
#                 segments.append((start * 0.01, len(pred_ids) * 0.01, prev))

#         return probs[np.newaxis, :, :], np.array(segments)

class charsiu_predictive_aligner_ko:
    def __init__(self, model_path, processor_path, vocab_path, device=None):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).eval()
        self.blank_id = self.processor.tokenizer.pad_token_id

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.id2label = {v: k for k, v in json.load(f).items()}
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _get_probs_and_preds(self, audio):
        if isinstance(audio, str):
            audio, sr = sf.read(audio)
            if sr != 16000:
                raise ValueError(f"Expected 16kHz sampling rate, got {sr}Hz")

        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, T, V)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_ids = np.argmax(probs, axis=-1)  # (T,)
        return probs, pred_ids

    def align(self, audio):
        """
        Greedy decoding - returns (start, end, phoneme_id) per frame
        """
        probs, pred_ids = self._get_probs_and_preds(audio)
        return np.array([(i * 0.01, (i + 1) * 0.01, p) for i, p in enumerate(pred_ids)])

    def align_probabilistic(self, audio, min_duration=3, merge_repeats=True):
        """
        CTC-style segmentation + top-k probs per segment
        Returns:
            - probs: (1, T, V)
            - segments: List[(start, end)]
        """
        probs, pred_ids = self._get_probs_and_preds(audio)
        segments = []
        prev = None
        start = 0

        for i, p in enumerate(pred_ids):
            if p == self.blank_id:
                continue
            if p != prev:
                duration = i - start
                if prev is not None and duration >= min_duration:
                    if merge_repeats and segments and prev == segments[-1][2]:
                        segments[-1] = (segments[-1][0], i * 0.01, prev)
                    else:
                        segments.append((start * 0.01, i * 0.01, prev))
                start = i
                prev = p

        if prev is not None and (len(pred_ids) - start) >= min_duration:
            if merge_repeats and segments and prev == segments[-1][2]:
                segments[-1] = (segments[-1][0], len(pred_ids) * 0.01, prev)
            else:
                segments.append((start * 0.01, len(pred_ids) * 0.01, prev))

        segs = [(s, e) for s, e, _ in segments]
        return probs[np.newaxis, :, :], np.array(segs)



