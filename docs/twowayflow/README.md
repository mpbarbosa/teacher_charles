# twowayflow — design projections of this project

This folder is the default home for everything produced when `teacher_charles` is explored
or analyzed with [twowayflow](https://github.com/mpbarbosa/twowayflow): diagrams,
`*.flow.ts` projections, and analysis notes.

Each file carries a titleblock naming the diagram, its type, the project and
what it shows, so it stays readable when pulled out of this repository.

## What is here

| File | What it projects |
|---|---|
| [`turn-pipeline.md`](turn-pipeline.md) | One turn of POST /api/chat as data dependencies: the student's utterance to the SSE frames and the rows SQLite keeps, with the review cycle that turns today's error into a later session's drill. |
