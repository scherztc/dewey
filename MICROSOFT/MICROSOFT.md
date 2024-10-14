# DEWEY can leverage PowerAutomation from Microsoft to determine the Title, Author, and Date of works submitted

1. Power Automate (https://powerautomate.microsoft.com/)
   1.  AI Builder
     1.  Models
     1.  Training Document
     1.  Information to Extract
   1. My Flows
     1.  Cloud Flows
         1.  Use form processing to extract data from documents triggered manually.
         1.  Flow Checker Feedback Flows and Run

   1. How to : https://learn.microsoft.com/en-us/microsoft-365/community/machine-learning-and-managed-metadata

# DEWEY can leverage Chat GPT (3.5, 3.5-turbo, 4.0, 4.0-=turbo) as a spell checker, voice response, or Image Creator :

https://github.com/openai/openai-cookbook


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

# References

# Regulation

MEPs substantially amended the list to include bans on intrusive and discriminatory uses of AI systems such as:


1.  “Real-time” remote biometric identification systems in publicly accessible spaces;
1.  “Post” remote biometric identification systems, with the only exception of law enforcement for the prosecution of serious crimes and only after judicial authorization;
1.  Biometric categorisation systems using sensitive characteristics (e.g. gender, race, ethnicity, citizenship status, religion, political orientation);
1.  Predictive policing systems (based on profiling, location or past criminal behaviour);
1.  Emotion recognition systems in law enforcement, border management, workplace, and educational institutions; and
1.  Indiscriminate scraping of biometric data from social media or CCTV footage to create facial recognition databases (violating human rights and right to privacy).

