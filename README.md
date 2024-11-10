# Amazon-ML-Challenge
We got 102 Rank with score of 0.54 over 2600+ teams.

#Problem Statement:
The objective of this hackathon is to build a machine learning solution that can derive critical information from images. This capability is essential for industries such as e-commerce, healthcare, and content management, where accurate product details are crucial. As online marketplaces grow, numerous products are displayed with limited descriptive text, making it important to identify specifics like weight, dimensions, wattage, and volume from images alone.

#Dataset: 
The dataset is divided into two main files:

train.csv: Contains over 310,000 image links along with metadata.
test.csv: Contains over 130,000 image links along with metadata.
Each dataset has the following columns:

index: An unique identifier (ID) for the data sample
image_link: Public URL where the product image is available for download. 
group_id: Category code of the product
entity_name: Product entity name. For eg: “item_weight”
entity_value: Product entity value. For eg: “34 gram” Note: For test.csv, you will not see the column entity_value as it is the target variable.

#Approach

As the Train dataset size is around 250k and for Test is around 130k , and as the images
given in the sample need to be downloaded , So first we downloaded the image
samples and store that in local after creating the metadata of the whole things.

• Then there could be two approaches which we decide to work that is one could be
fine-tuning the Multi- Modal LLM like Open-AI Clip , PaliGemma , Llava Qwen2vl but it takes too much time to finetune multimodal LLM as we have huge dataset it requires atleast 60 - 90 GB+ of atleast 24 Hours of T4 GPU which is very costly And  another approach which could be  extracting the Text from the Images using PaddleOCR or Easy OCR  and then  Finetune it with  any LLM like Gemma 2 , Mistral , LLama others.

• But we got less resources to fine-tune any multi-modal LLM , It takes almost 1 hours to
tune with only 1000 samples so we finalise over approach to extract the captions or Texts and
fine-tune it with Mistral-Nemo-12 Billion Parameters and  Gemma2- 9Billion .

• We used Paddle-OCR to extract all the image captions of train and test set then make
that saved in side the metadata_df with extracted - text.

• Then we did LORA(Low Rank Adapters) Fine-tuning of the Mistral-Nemo model and Gemma2-9B using Unsloth FastTrainedModel and
downloaded the model which is then tuned with these model parameters :

<img width="855" alt="Screenshot 2024-11-11 at 12 27 17 AM" src="https://github.com/user-attachments/assets/a54fad6c-ab23-415f-8c13-e1483d66b16b">

And the using SFT(Supervised Finetuning) trainer from TRL we just passed the arguments below and tuned it :

<img width="850" alt="Screenshot 2024-11-11 at 12 30 14 AM" src="https://github.com/user-attachments/assets/21ef367a-a210-4ec7-875b-aabe8d3b4fb3">

Then save the safetensors , tokenisers and configs .
• Then For inference we load the tuned model and passed the test set with caption
extracted text using Paddle -OCR and save the submission.csv file , which is used to
submit the inference.

#Results:
Our Model Achieved Following score :
F1 score : 0.54
Rank : 102 out of 2600+ teams

