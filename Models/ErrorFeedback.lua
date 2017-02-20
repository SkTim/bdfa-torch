local ErrorFeedback, Parent = torch.class('ErrorFeedback', 'nn.Module')

function ErrorFeedback:__init(magnitude)
  Parent.__init(self)
  self.feedback = torch.Tensor()
  self.feedforward = torch.Tensor()
  -- self.gradWeight = torch.Tensor()
  self.buffer = torch.Tensor()
  self.mag = magnitude or 0
  self.source_buffer = torch.Tensor()
  self.input_buffer = torch.Tensor()
  self.predict_buffer = torch.Tensor()
  -- self.batch_normlization = nn.BatchNormalization(opt.num_hidden):cuda()
end

function ErrorFeedback:updateOutput(input)

  local nElement = self.output:nElement()
  self.output:resizeAs(input)
  if self.output:nElement() ~= nElement then
    self.output:zero()
  end
  
  self.output:copy(input)

  return self.output
end

function ErrorFeedback:updateGradInput(input, gradOutput)
  local nElement = self.gradInput:nElement()
  self.gradInput:resizeAs(input)
  if self.gradInput:nElement() ~= nElement then
    self.gradInput:zero()
  end

  nElement = self.feedback:nElement()
  nElement2 = self.feedforward:nElement()
  if input:dim() == 4 then
    self.feedback:resize(gradOutput:size(2), input:size(2)*input:size(3)*input:size(4))
    -- self.gradWeight:resize(gradOutput:size(2), input:size(2)*input:size(3)*input:size(4))
  elseif input:dim() == 3 then
    self.feedback:resize(gradOutput:size(2), input:size(2)*input:size(3))
    -- self.gradWeight:resize(gradOutput:size(2), input:size(2)*input:size(3))
  else
    self.feedback:resize(gradOutput:size(2), input:size(2))
  end

  self.feedforward:resize(3072, 10)

  --[[
  if input:dim() == 4 then
    self.feedforward:resize(3072, input:size(2)*input:size(3)*input:size(4))
    -- self.gradWeight:resize(gradOutput:size(2), input:size(2)*input:size(3)*input:size(4))
  elseif input:dim() == 3 then
    self.feedforward:resize(3072, input:size(2)*input:size(3))
    -- self.gradWeight:resize(gradOutput:size(2), input:size(2)*input:size(3))
  else
    self.feedforward:resize(3072, input:size(2))
    -- self.gradWeight:resize(gradOutput:size(2), input:size(2))
  end
  ]]--
  
  if self.feedback:nElement() ~= nElement then
    if self.mag == 0 then
      self.mag = 1/math.sqrt(self.feedback:size(2))
    end
    self.feedback:uniform(-self.mag, self.mag)
  end

  if self.feedforward:nElement() ~= nElement2 then
    if self.mag == 0 then
      self.mag = 1/math.sqrt(self.feedforward:size(2))
    end
    self.feedforward:uniform(-self.mag, self.mag)
  end
  
  if input:dim() == 4 then
    self.buffer:resize(input:size(1), input:size(2)*input:size(3)*input:size(4))
  elseif input:dim() == 3 then
    self.buffer:resize(input:size(1), input:size(2)*input:size(3))
  else
    self.buffer:resize(input:size(1), input:size(2))
  end

  self.buffer:zero()
  if gradOutput:dim() == 1 then
    torch.mv(self.buffer, self.feedback:t(), gradOutput)
  elseif gradOutput:dim() == 2 then
    torch.mm(self.buffer, gradOutput, self.feedback)
  end
  
  self.gradInput:zero()
  self.gradInput:add(self.buffer:view(input:size()))
  
  return self.gradInput
end

function label_matrix(labels, n_classes)
    n = labels:size(1)
    m = torch.zeros(n, n_classes):cuda()
    for i = 1, n do
        m[i][labels[i]] = 1
    end
    return m
end

function gsigmoid(x)
  return torch.cmul(x, 1 - x)
end

function dtanh(x)
  return 1 - torch.cmul(x, x)
end

-- --[[
function ErrorFeedback:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   self.input_buffer:resizeAs(input)
   self.input_buffer:copy(input)
   -- self.source_buffer:resizeAs(self.x)
   -- self.source_buffer:copy(self.x)
   if input:dim() == 4 then
     self.input_buffer:resize(input:size(1), input:size(2)*input:size(3)*input:size(4))
   elseif input:dim() == 3 then
     self.input_buffer:resize(input:size(1), input:size(2)*input:size(3))
   else
     self.input_buffer:resize(input:size(1), input:size(2))
   end
   
   -- if self.x:dim() == 4 then
   --   self.source_buffer:resize(self.x:size(1), self.x:size(2)*self.x:size(3)*self.x:size(4))
   -- elseif self.x:dim() == 3 then
   --  self.source_buffer:resize(self.x:size(1), self.x:size(2)*self.x:size(3))
   -- else
   --   self.source_buffer:resize(self.x:size(1), self.x:size(2))
   -- end

   -- self.predict_buffer:resizeAs(self.input_buffer)
   -- print(self.feedback:size())
   local labels = label_matrix(self.yt, 10) -- :add(self.y:mul(0.1))
   -- :add(torch.mm(self.source_buffer, self.feedforward):mul(0.1))
   self.predict_buffer = torch.mm(labels, self.feedback)

   -- self.predict_buffer:add(torch.sigmoid(self.source_buffer, self.feedforward))

   -- torch.mm(self.predict_buffer, self.yt, self.feedback)
   -- self.predict_buffer = torch.sigmoid(self.predict_buffer)
   self.predict_buffer = torch.tanh(self.predict_buffer)
   -- self.predict_buffer:csub(self.input_buffer)
   local gradient = torch.csub(self.predict_buffer, self.input_buffer)
   -- local dt = dtanh(self.predict_buffer)
   local dt = dtanh(torch.cmul(self.predict_buffer, gradient))
   self.feedback:csub(0.00005 * torch.mm(labels:t(), dt))

   -- self.feedforward:csub(0.00005 * torch.mm(self.source_buffer:t(), dtanh))
   
end

function ErrorFeedback:sharedAccUpdateGradParameters(input, gradOutput, lr)
   -- we do not need to accumulate parameters when sharing:
   self:defaultAccUpdateGradParameters(input, gradOutput, lr)
end
-- ]]--

function ErrorFeedback:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.mag)
end

