require 'openai-client'

Openai::Client.configure do |c|
  c.access_token    = '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'
  c.organization_id = 'organization_id' # optional
end
