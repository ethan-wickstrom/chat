defineRouteMeta({
  openAPI: {
    description: 'List the current user reasoning traces stored in Neo4j.',
    tags: ['memory']
  }
})

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)
  const query = getQuery(event)
  const rawLimit = Array.isArray(query.limit) ? query.limit[0] : query.limit
  const requestedLimit = Number(rawLimit || 25)
  const limit = Number.isFinite(requestedLimit)
    ? Math.min(Math.max(Math.trunc(requestedLimit), 1), 100)
    : 25
  const rawSuccess = Array.isArray(query.success) ? query.success[0] : query.success
  const success = rawSuccess === 'true' ? true : rawSuccess === 'false' ? false : null

  return listReasoningTraces({
    userId: session.user?.id || session.id,
    limit,
    success
  })
})
