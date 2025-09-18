Context Cache on Vertex

- Keep a stable prompt prefix (policy/methods/style) to maximize implicit cache hits.
- Cached input tokens receive roughly a 75% discount, per Vertex docs.
- Our services set the X-Context-Cache: ENABLED header when applicable.

Guidelines

- Place shared instructions at the beginning.
- Keep dynamic parts (user query/data) in a trailing suffix section.
- Avoid frequent changes in the prefix to preserve cache keys.

