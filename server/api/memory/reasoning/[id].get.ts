defineRouteMeta({
  openAPI: {
    description: 'Get one reasoning trace with its step and tool-call path.',
    tags: ['memory']
  }
})

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)
  const { id } = getRouterParams(event)

  const trace = await getReasoningTrace(id as string, session.user?.id || session.id)
  if (!trace) {
    throw createError({ statusCode: 404, statusMessage: 'Reasoning trace not found' })
  }

  return trace
})
