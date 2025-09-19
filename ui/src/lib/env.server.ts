import 'server-only'
import { z } from 'zod'

const serverSchema = z.object({
  NEXT_PUBLIC_DARWIN_URL: z.string().url(),
  DARWIN_SERVER_KEY: z.string().min(1),
})

export function getServerEnv() {
  return serverSchema.parse({
    NEXT_PUBLIC_DARWIN_URL: process.env.NEXT_PUBLIC_DARWIN_URL,
    DARWIN_SERVER_KEY: process.env.DARWIN_SERVER_KEY,
  })
}
