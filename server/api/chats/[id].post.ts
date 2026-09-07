import { convertToModelMessages, createUIMessageStream, createUIMessageStreamResponse, generateText, streamText } from 'ai'
import { gateway } from '@ai-sdk/gateway'
import type { UIMessage } from 'ai'
import { z } from 'zod'

defineRouteMeta({
  openAPI: {
    description: 'Chat with AI.',
    tags: ['ai']
  }
})

function getMessageText(message: UIMessage | undefined) {
  if (!message) {
    return ''
  }

  return message.parts.map((part) => {
    if (part.type !== 'text') {
      return ''
    }

    return 'text' in part && typeof part.text === 'string' ? part.text : ''
  }).filter(Boolean).join('\n').trim()
}

function getErrorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message
  }

  if (typeof error === 'string') {
    return error
  }

  try {
    return JSON.stringify(error)
  } catch {
    return 'Unknown stream error'
  }
}

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  const { id } = getRouterParams(event)

  const { model, messages } = await readValidatedBody(event, z.object({
    model: z.string(),
    messages: z.array(z.custom<UIMessage>())
  }).parse)

  const db = useDrizzle()

  const chat = await db.query.chats.findFirst({
    where: (chat, { eq }) => and(eq(chat.id, id as string), eq(chat.userId, session.user?.id || session.id)),
    with: {
      messages: true
    }
  })
  if (!chat) {
    throw createError({ statusCode: 404, statusMessage: 'Chat not found' })
  }

  if (!chat.title) {
    const { text: title } = await generateText({
      model: gateway('openai/gpt-4o-mini'),
      system: `You are a title generator for a chat:
          - Generate a short title based on the first user's message
          - The title should be less than 30 characters long
          - The title should be a summary of the user's message
          - Do not use quotes (' or ") or colons (:) or any other punctuation
          - Do not use markdown, just plain text`,
      prompt: JSON.stringify(messages[0])
    })

    await db.update(tables.chats).set({ title }).where(eq(tables.chats.id, id as string))
  }

  const lastMessage = messages[messages.length - 1]
  if (lastMessage?.role === 'user' && messages.length > 1) {
    await db.insert(tables.messages).values({
      chatId: id as string,
      role: 'user',
      parts: lastMessage.parts
    })
  }

  const userId = session.user?.id || session.id
  const task = getMessageText(lastMessage) || 'Continue the conversation'
  const { traceId, recalled } = await beginReasoningTrace({
    sessionId: chat.id,
    userId,
    task,
    model,
    recallLimit: 3
  })
  const recalledContext = formatRecalledReasoning(recalled)
  const system = [
    'You are a helpful assistant that can answer questions and help.',
    recalledContext
  ].filter(Boolean).join('\n\n')

  let stepIndex = 0
  let streamError: string | null = null

  const stream = createUIMessageStream({
    execute: ({ writer }) => {
      const result = streamText({
        model: gateway(model),
        system,
        messages: convertToModelMessages(messages),
        onStepFinish: async ({ finishReason, text, toolCalls, toolResults, usage }) => {
          await recordReasoningStep(traceId, stepIndex++, {
            finishReason,
            text,
            toolCalls,
            toolResults,
            usage
          })
        },
        onError: ({ error }) => {
          streamError = getErrorMessage(error)
          void completeReasoningTrace(traceId, {
            success: false,
            finishReason: 'error',
            errorKind: 'stream_error',
            error: streamError
          })
        },
        onFinish: async ({ text, finishReason }) => {
          await completeReasoningTrace(traceId, {
            success: streamError === null && finishReason !== 'error',
            finishReason,
            responseText: text,
            errorKind: streamError ? 'stream_error' : null,
            error: streamError
          })
        }
      })

      if (!chat.title) {
        writer.write({
          type: 'data-chat-title',
          data: { message: 'Generating title...' },
          transient: true
        })
      }

      writer.merge(result.toUIMessageStream())
    },
    onFinish: async ({ messages }) => {
      await db.insert(tables.messages).values(messages.map(message => ({
        chatId: chat.id,
        role: message.role as 'user' | 'assistant',
        parts: message.parts
      })))
    }
  })

  return createUIMessageStreamResponse({
    stream
  })
})
