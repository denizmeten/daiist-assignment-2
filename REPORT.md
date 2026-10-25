# Assignment 2 Report

- **Name:** Deniz Meten
- **Student ID:** 18758
- **Email:** dmeten.ieu2022@student.ie.edu
- **Group:** BBADBA 5A

## Business problem, and the cost of being wrong

For this project, I wanted to use computer vision for something related to gaming because it is a domain I understand well. The system is designed to support Valorant VOD and replay analysis.

The main decision is whether a gameplay frame contains a visible enemy body and therefore should be considered relevant for review. Instead of a coach or analyst manually going through every sampled frame of a VOD, the model can automatically skip low-confidence frames, flag high-confidence frames, and send uncertain cases to a person.

The positive business class is `enemyBody`.

I treated a false negative as more expensive than a false positive. A false negative means that there actually is an enemy visible but the system misses the frame. This can cause an important combat or positioning situation to be skipped completely and may require the analyst to search through the VOD manually later.

A false positive means that the system flags a frame even though there is no enemy body. This is still a mistake, but the main consequence is that the analyst wastes some time checking an unnecessary frame.

For the cost model, I used an analyst cost of €48 per hour.

- False positive: around 15 seconds of unnecessary review = **€0.20**
- False negative: around 120 seconds of additional manual searching = **€1.60**
- Human review of an uncertain frame: around 15 seconds = **€0.20**

This means a false negative is treated as eight times more expensive than a false positive. These numbers are assumptions rather than values provided by the dataset, but the reasoning is that missing a useful combat situation is more costly than checking one unnecessary frame.

## Dataset

I used a public object-detection dataset from Roboflow Universe.

- **Workspace:** `kwan-li-jqief`
- **Project:** `valorant-object-detection2`
- **Version:** `7`
- **Task:** Object detection
- **Format:** YOLOv11
- **Classes:** `enemyBody` and `enemyHead`

The dataset was used directly from Roboflow Universe and was not forked or relabelled.

One image represents a Valorant gameplay frame. Bounding boxes identify visible enemy bodies and enemy heads.

Version 7 contains **1,570 images** in total.

The dataset split was:

- Training: **1,416 images**
- Validation: **103 images**
- Test: **51 images**

The number of annotations per class was:

| Split | enemyBody | enemyHead |
|---|---:|---:|
| Train | 1,243 | 1,254 |
| Validation | 87 | 87 |
| Test | 47 | 42 |
| **Total** | **1,377** | **1,383** |

The classes are therefore very balanced overall.

Object size was another important part of the dataset. The median `enemyBody` box covered around **0.64% of the image**, while the median `enemyHead` box covered only around **0.15%**.

This means heads are much smaller targets than bodies, which makes them harder to detect reliably.

The biggest weakness of the dataset is the small test split. With only 51 test images, a few extra correct or incorrect predictions could noticeably change the final results. Because of this, the test results should not be treated as a perfect estimate of performance across every possible Valorant match.

## Finetuning and baselines

I used **YOLO11n** for object detection.

I selected the nano version because the dataset is not extremely large and the model is lightweight enough to train and deploy efficiently while still benefiting from pretrained YOLO features.

The final model was trained for **50 epochs** using an image size of **640** and a batch size of **16**. I trained the model locally using an **NVIDIA GeForce RTX 5070 Ti Laptop GPU**.

The model was compared against two baselines on the same held-out test split:

1. Finetuned YOLO11n trained on the Valorant dataset.
2. Pretrained YOLO11n using COCO weights without Valorant finetuning.
3. A trivial baseline that predicts nothing.

For the COCO zero-shot comparison, the generic COCO `person` class was mapped to `enemyBody`, because COCO does not contain the Valorant-specific `enemyBody` and `enemyHead` classes. There is no direct COCO equivalent for `enemyHead`.

The results were:

| Model | Main result |
|---|---|
| Finetuned YOLO11n | Precision 0.770, Recall 0.792, mAP50 0.790 |
| COCO YOLO11n zero-shot | 0.00 precision and 0.00 recall for the mapped Valorant classes at threshold 0.50 |
| Predict-nothing baseline | 0.00 recall because it never predicts a positive detection |

For the COCO zero-shot model, the mapped `person` detections produced **37 false positives** for `enemyBody`, but none matched the 47 ground-truth `enemyBody` objects correctly at the required IoU threshold. This resulted in:

- True positives: 0
- False positives: 37
- False negatives: 47
- Precision: 0.00
- Recall: 0.00

For `enemyHead`, the zero-shot baseline also had 0 precision and 0 recall because COCO has no direct equivalent for this class.

This shows that generic COCO object detection is not enough for this specific visual problem. Finetuning was necessary for the model to learn the Valorant-specific appearance and annotation style.

The finetuned model achieved:

| Metric | Result |
|---|---:|
| Precision | 0.770 |
| Recall | 0.792 |
| mAP50 | 0.790 |
| mAP50-95 | 0.297 |

## Per-class evaluation and failure modes

The final per-class test results were:

| Class | Precision | Recall | AP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| enemyBody | 0.7800 | 0.8298 | 0.8054 | 0.2843 |
| enemyHead | 0.7601 | 0.7546 | 0.7747 | 0.3095 |

`enemyBody` had the strongest AP50 and recall. This is useful because `enemyBody` is also the positive class used for the business decision.

`enemyHead` had lower recall, meaning that the model missed heads more often than bodies. One likely reason is object size. The median head annotation occupies only around 0.15% of the image compared with around 0.64% for a body.

The difference between mAP50 and mAP50-95 is also important. The model performs well when the required bounding-box overlap is moderate, but performance drops when stricter localization is required. This suggests that the model often identifies the correct object but does not always place the bounding box extremely precisely.

## Cost matrix and operating point

The model outputs a confidence score for its detections. Instead of automatically using a threshold of 0.50, I swept the confidence threshold from 0.01 to 0.99.

For the model-only decision, the cost function was:

`Total cost = FP × €0.20 + FN × €1.60`

The observed positive base rate in the held-out test set was **0.843**, meaning approximately 84.3% of the test images contained at least one `enemyBody`.

Because the positive class is very common and false negatives are much more expensive than false positives, the cost model strongly prefers a low confidence threshold.

The minimum-cost model-only threshold was:

**0.01**

This is much lower than a typical 0.50 threshold, but it follows from the assumptions. The model is encouraged to make more positive predictions because incorrectly flagging a frame costs only €0.20, while completely missing an enemy frame is assumed to cost €1.60.

The dashboard cost curve shows that expected cost increases as the threshold becomes more conservative.

The most important assumption affecting this result is the false-negative cost. If missing an enemy were much less expensive than assumed, the optimal confidence threshold would move higher.

## What a person still has to check

Instead of using only one confidence threshold, I also implemented a three-outcome system:

- Score below **0.05** → auto-pass / skip
- Score from **0.05 to 0.10** → human review
- Score of **0.10 or above** → auto-flag

The selected assisted review band was:

**0.05 – 0.10**

At this operating point, the test-set results were:

- True positives: **43**
- False positives: **3**
- True negatives: **5**
- False negatives: **0**
- Human-review share: **2.0%**
- Expected cost per 1,000 cases: **€15.69**

Assuming **2,000 sampled frames per day**, around **39.2 frames per day** would be sent to human review.

At approximately 15 seconds per review, this equals around **0.16 hours of review work per day**.

This is a small workload compared with checking every frame manually.

The main advantage of the assisted system is that it does not force the model to make every decision by itself. Highly confident frames can be handled automatically, while uncertain frames can still be escalated to a person.

The zero false negatives in this specific test result should not be interpreted as proof that the model will never miss an enemy in production. The test split only contains 51 images, so this result is based on a relatively small sample.

Compared with the alternatives:

**Doing nothing:** analysts continue manually searching through complete VOD footage.

**Human review of everything:** gives maximum human control but creates much more manual workload.

**Model-only automation:** minimizes human effort but increases the risk that uncertain cases are handled incorrectly.

**Model plus human review:** provides a middle ground by automatically handling confident cases and sending only a small uncertain group to a person.

## Limitations & next steps

The first limitation is the small test split. With only 51 images, the reported performance and the chosen thresholds could change significantly if a larger test set was used.

The second limitation is that the dataset may not represent every possible Valorant situation. Different maps, agents, weapons, abilities, resolutions, graphics settings and recording quality may affect model performance.

Another limitation is that the model only detects visual objects. Detecting an enemy does not automatically mean a frame is tactically important. A real coaching tool would also need information such as player position, map location, round state, economy, utility usage and what happens before and after the frame.

The production base rate is also uncertain. I used the test-set base rate of 84.3% because no separate production stream was available. A real VOD review system could have a very different proportion of frames containing enemies, and this could change the optimal threshold.

The business costs are also assumptions based on analyst time rather than measured organizational data.

With more time, I would evaluate the model on a larger and more varied set of Valorant recordings, measure the real production base rate and analyst review times, and then recalculate the cost-optimal thresholds using those real values.

## Generative AI use disclosure

Generative AI was used to assist with boilerplate code structure, Roboflow and YOLO integration, evaluation code, debugging, threshold-sweep implementation, dashboard wiring, and report formatting.

The business assumptions, interpretation of the results, operating decision, and final conclusions were reviewed by me, and I understand the submitted pipeline and the reasoning behind the chosen thresholds.
