# Scene_Finder

![Scene Finder Example](https://github.com/hyokyunAn/scene_finder/assets/60477870/631f57f6-5719-4545-880c-56ccd544dfc3)

- **Example Video:** [Watch here](https://www.youtube.com/watch?v=z4K2F_OALPQ&t=40s)
- **Example Prompt:** A hat shouting "Gryffindor!" on a girl's head.

[Demo](http://34.95.2.121:5000/)
(Demo sometimes works badly due to status of hosting server or API server (like OpenAI GPT)

## How It Works?

1. **Convert the scene description** into descriptions for the image and the dialogue using the fine-tuned GPT 3.5 model.
   - *Original Prompt:* A hat shouting "Gryffindor!" on a girl's head.
   - *Transformed Prompt:* 
     ```json
     {
       "image_prompt": "A hat shouting",
       "text_prompt": "Gryffindor!"
     }
     ```
2. **Extract images** whenever there's a cut in the video (Shot detection).
3. **Extract characters' dialogues** (Google Speech-to-Text).
4. **Measure the similarity** of the extracted images to the image_prompt (OpenAI CLIP).
5. **Find the most similar dialogue** from the extracted dialogues to the text_prompt (Sentence BERT).
6. **Combination:** Combine steps 4 and 5 to select one image, choose one image from step 4, and another image from step 5, resulting in a total of three images. Then, return the times each image appears.

## Models

- **Fine-tuned ChatGPT:** [Documentation](https://platform.openai.com/docs/guides/fine-tuning)
- **OpenAI CLIP:** [Research Link](https://openai.com/research/clip)
- **Google Speech-to-Text:** [Product Page](https://cloud.google.com/speech-to-text?hl=ko#put-speech-to-text-into-action)
- **Sentence BERT:** [Model Link](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Requirements

- `OPENAI_API_KEY`
- `GOOGLE_CLOUD_API_KEY` (JSON format)
- `GOOGLE_CLOUD_STORAGE`

## Acknowledgements

This project was inspired by these projects:

- **Unsplash Image Search:** [GitHub Repository](https://github.com/haltakov/natural-language-image-search)
- **alpaca:** [Model Page](https://huggingface.co/chainyo/alpaca-lora-7b)

