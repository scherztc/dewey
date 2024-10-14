# DEWEY can leverage Chat GPT (3.5, 3.5-turbo, 4.0, 4.0-=turbo) as a spell checker, voice response, or Image Creator :

1.   Chat GPT (OpenAI) : https://openai.com/blog/chatgpt
   -  Github : https://github.com/openai
   -  Explore : https://github.com/openai/openai-cookbook
   -  Uses the openai-client: https://github.com/itikhonenko/openai-client
      -  API KEY
      -  ORGANIZATION_ID
   -  Dall-E (OpenAI) : https://openai.com/product/dall-e-2

   1.  Spell Checker
 
         request_body = {
            model: 'text-davinci-edit-001',
            input: 'What day of the wek is it?',
         instruction: 'Fix the spelling mistakes'
         }
         Openai::Client.edits.create(request_body)

   1.  Image Creator

         request_body = {
            prompt: 'A cute baby sea otter',
            n: 1,                  # between 1 and 10
            size: '1024x1024',     # 256x256, 512x512, or 1024x1024
            response_format: 'url' # url or b64_json
         }
      
        response = Openai::Client.images.create(request_body)

   1. Connect in Ruby

        require 'openai-client'

        Openai::Client.configure do |c|
          c.access_token    = 'access_token'
          c.organization_id = 'organization_id' # optional
        end

   1. Find Engine

        Openai::Client.models.find(‘babbage’)
        Openai::Client.models.find(‘davinci’)

   1. Build Request Body

        request_body = {
           prompt: 'high swim banquet',
           n: 1,                  # between 1 and 10
           size: '1024x1024',     # 256x256, 512x512, or 1024x1024
           response_format: 'url' # url or b64_json
        }

    1. Playground interface : https://platform.openai.com/playground?mode=chat

#  DEWEY can leverage Machine Learning and Large Language Models

    1.  Repository : https://huggingface.co/
    1.  The Bloke : https://huggingface.co/TheBloke
    1.  Lone Striker : https://huggingface.co/LoneStriker
    1.  WebGUI : https://github.com/oobabooga/text-generation-webui
    1.  Stable Diffusion : https://github.com/AUTOMATIC1111/stable-diffusion-webui
    1.  Voice Changer : github.com/w-okada/voice-changer
    1.  Real Time Voice : https://github.com/RVC-Project/Retrieval-based-VOice-Conversion-WebUI
    1.  RVC : voice-models.com  and weighs.gg

#  Other AI Engines to Explore
  
1.  Stable Diffusion (Stability) : https://stablediffusionweb.com/  or civitai.com
1.  Watson (IBM) : https://www.ibm.com/products/watson-explorer 
   1. Chess
   1. Content Hub. IBM Watson can propose relevant tags based on content.
1.  Bard/Palm 2 (Google)
   1. Google blog post about BERT,18 an ML technique for NLP, the benefit shown was simply the ability to link a preposition with a noun. 
1.  Aladin (BlackRock)
1.  Mindjourney (MindJourney) : https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F
1.  Kaaros
1.  Tensor Flow (Google)
1.  IRIS : https://iris.ai/      
1.  Claude https://www.anthropic.com/index/claude-2
1.  https://marketplace.atlassian.com/apps/1224655/scrum-maister?hosting=cloud&tab=overview
1.  Bing (free)
1.  Claude 2 (free) by Anthropic
1.  Grok by X (Twitter)
1.  Open-source models (FREE) available on Huggingface https://huggingface.co/
1.  Llama 2 by Meta
1.  Flan, Falcon, Orca, Beluga, Mistral, Mixtral, Phi2
1.  LMStudio (Windows, Mac, Linux) - install and run models
1.  Pinokio.computer browser - install and run models
1.  Atlassian Rovo - https://www.atlassian.com/blog/announcements/introducing-atlassian-rovo-ai

# References

# Regulation

MEPs substantially amended the list to include bans on intrusive and discriminatory uses of AI systems such as:


1.  “Real-time” remote biometric identification systems in publicly accessible spaces;
1.  “Post” remote biometric identification systems, with the only exception of law enforcement for the prosecution of serious crimes and only after judicial authorization;
1.  Biometric categorisation systems using sensitive characteristics (e.g. gender, race, ethnicity, citizenship status, religion, political orientation);
1.  Predictive policing systems (based on profiling, location or past criminal behaviour);
1.  Emotion recognition systems in law enforcement, border management, workplace, and educational institutions; and
1.  Indiscriminate scraping of biometric data from social media or CCTV footage to create facial recognition databases (violating human rights and right to privacy).

