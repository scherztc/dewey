# Chat GPT (3.5, 3.5-turbo, 4.0, 4.0-=turbo) as a spell checker, voice response, or Image Creator :

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

