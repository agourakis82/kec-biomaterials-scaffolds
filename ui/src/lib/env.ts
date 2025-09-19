import { z } from 'zod'

// Client-only env: do not import server secrets here.
const clientSchema = z.object({
  NEXT_PUBLIC_DARWIN_URL: z.string().url(),
})

export const clientEnv = clientSchema.parse({
  NEXT_PUBLIC_DARWIN_URL: process.env.NEXT_PUBLIC_DARWIN_URL,
})
