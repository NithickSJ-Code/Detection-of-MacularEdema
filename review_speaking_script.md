# Review Presentation Script
*Each person's part is roughly 2-3 minutes. Total presentation: ~10 minutes.*

---

## NITHIN SB — Opens the presentation
*(Slides: Title, Abstract, Introduction)*

**[Title Slide]**
"Good morning ma'am. We are Team 05. Our project is on Early Detection of Macular Edema in OCT scans using Deep Learning with Grad-CAM Localization. I'll start by giving an overview of what the project is about."

**[Abstract Slide]**
"So Macular Edema is basically a condition where fluid starts leaking into the center of the retina — the macula — and if it's not caught early, the patient loses their central vision permanently. Doctors diagnose this using OCT scans, which are like cross-section images of the eye. The problem is, manually going through thousands of these scans is slow and depends entirely on how experienced the doctor is."

"Most AI solutions for this use something called a U-Net, which needs pixel-perfect masks drawn by doctors on every single training image. That data is almost impossible to get. So our approach is different — we're using a classification model combined with a technique called Grad-CAM to locate the fluid without needing any of those hand-drawn masks."

**[Introduction Slide]**
"To give a bit more context — when fluid builds up in the macula, it creates these small dark pockets in the OCT scan. A trained ophthalmologist can spot these, but with the volume of patients in a hospital like Aravind Eye Hospital, it's not realistic to manually check every scan. That's the gap our system is trying to fill."

*"I'll hand it over to Nishant to explain the technical side."*

---

## NISHANT J S — The Technical Architecture
*(Slides: Existing System, Drawbacks, Proposed System, Technologies Used)*

**[Existing System Slide]**
"Right now, the standard way AI researchers solve this is by using 'segmentation'. Basically, a doctor has to sit down and manually draw outlines around every single pocket of fluid in thousands of scans. Once they have all these drawn masks, they train a model to try and copy that behavior on new images."

**[Drawbacks Slide]**
"But there are some major problems with that. First, doctors are busy — getting them to manually draw masks on 80,000 images is practically impossible, so that kind of data is super rare and expensive. Second, standard AI models are often way too confident. Even if you feed them a blurry, messed up scan, they might say 'I am 99% sure this patient is fine', which is obviously super dangerous in real-world hospitals."

**[Proposed System Slide]**
"So, our solution skips the hand-drawn masks entirely. We train a model to simply classify whether a scan has a disease or not. But here is the cool part: to figure out *where* the disease actually is, we use a technique called Grad-CAM. It essentially tracks where the AI's 'eyes' were looking when it made its decision. We then use that to draw a bounding box around the fluid. And to fix the overconfidence issue, we test the same image 30 different times with random parts of the model turned off. If the model starts giving us different answers, we know it's confused and we flag it for a real doctor to check."

**[Technologies Used Slide]**
"To build this, we're mostly using Python. The core AI is built with PyTorch using a model called DenseNet-121. We use OpenCV for image processing, and Grad-CAM for the heatmaps. Later on, we plan to wrap all this up into a simple website."

*"Nithick will walk through the literature survey."*
---

## Nithick — Literature Survey & Flow Diagram
*(Slides: Literature Survey x3, Flow Diagram)*

**[Literature Survey Slides]**
"We looked at three key papers that directly relate to what we're building."

"The first one is Kermany et al., 2018 — this is the paper that released the OCT dataset we're using. They used transfer learning with InceptionV3 and showed that deep learning can match ophthalmologist-level accuracy on OCT classification. But their model only classifies — it doesn't tell you where the disease is."

"The second paper is Seebock et al., 2020 — they worked on weakly supervised anomaly detection in OCT using autoencoders and Grad-CAM. This is closest to what we're doing. They showed that you can get spatial information from a classification model without segmentation masks. But their bounding box accuracy on small lesions was still limited."

"The third one is Wang et al., 2022 — they focused on uncertainty estimation in medical image classification using Monte Carlo Dropout and Bayesian CNNs. They showed that tracking prediction variance across multiple stochastic forward passes gives you a reliable measure of model confidence. This is exactly the approach we adopted for our clinical safety layer."

**[Flow Diagram Slide]**
"This is the overall flow of our system. The OCT scan comes in, gets preprocessed and augmented, passes through the DenseNet-121 which outputs a classification. Then Grad-CAM generates a heatmap from the final convolutional layer, we threshold it to get bounding boxes, and simultaneously we run MC Dropout to get an uncertainty score. The final output is the diagnosis, the localization, and the confidence level — all three together."

*"Nishant will cover the results."*

---

## NISHANT JR — Results, Output, Conclusion
*(Slides: Results & Analysis, Output screenshots, Conclusion, References)*

**[Results Slide]**
"So where are we at right now? We've successfully trained our classification model on a massive dataset of over 80,000 images, and we've got the localization heatmaps working perfectly. When we tested the system with brand new images, the model showed a 100% accuracy in correctly identifying the disease, and more importantly, it knew exactly where to look."

**[Output Slides]**
"If you look at the screen, these are the actual real-time outputs generated by our model. We show three things: On the far left is the original OCT eye scan. In the middle is the Grad-CAM heatmap — the bright red areas show exactly what the AI was staring at to make its diagnosis. And on the right, we draw a clear bounding box over the fluid region. It completely removes the guesswork for the doctor."

*(flip through the 6 output slides)*
"As you can see across these different cases, whether it's fluid from DME, CNV, or just a Normal scan, the model successfully identifies it and highlights the correct region without needing any manual marking from us."

**[Conclusion Slide]**
"To wrap it up — we've built a system that doesn't just give a blind 'yes or no' answer. It tells you what the disease is, it points to exactly where the problem is, and it tells you if it's unsure. And we did it all without needing those expensive, hand-drawn masks. Going forward into the next review, we'll be turning this into a simple web application so anyone can easily upload a scan and get these results in real-time."

**[References Slide]**
"These are the research papers we referred to. Thank you ma'am."

---

## HANDLING QUESTIONS (Quick cheat sheet for all 4 of you)

**"Why DenseNet and not ResNet?"**
→ DenseNet connects every layer to every other layer, so gradients flow better and features get reused. It's more parameter-efficient than ResNet for medical imaging where the features are subtle.

**"What is Grad-CAM exactly?"**
→ After the model predicts "Edema," we compute the gradients flowing into the last convolutional layer to see which spatial regions contributed most to that prediction. We average those gradients across channels, weight the feature maps, apply ReLU, and overlay as a heatmap.

**"How is MC Dropout different from normal Dropout?"**
→ Normal Dropout is only active during training. We keep it active during testing. We run the image 30 times, get 30 slightly different answers because different neurons are turned off each time, and check if they agree or disagree. Disagreement = uncertainty.

**"What dataset are you using?"**
→ Kermany OCT dataset from Kaggle — over 83,000 retinal OCT images across 4 classes: Normal, DME (Diabetic Macular Edema), CNV, and Drusen.

**"Why not just use a U-Net?"**
→ U-Net needs pixel-level ground truth masks. That data is proprietary and locked behind hospital agreements. Our approach gets localization for free from the classification model using Grad-CAM.
