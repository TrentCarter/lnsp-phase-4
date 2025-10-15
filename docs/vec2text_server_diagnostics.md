# Vec2Text Server Diagnostics Cheatsheet

Use these helpers to confirm the decoder stack is healthy.

## Configuration Snapshot

```bash
curl -s http://localhost:8766/config | jq
```

Returns the loaded decoders, runtime devices, and whether a Procrustes adapter is active.

## Self-Test Probe

```bash
curl -s http://localhost:8766/selftest \
  -H 'Content-Type: application/json' \
  -d '{"text":"Photosynthesis converts light energy to chemical energy in plants.","steps":1}' \
  | jq
```

This validates the teacher-space round-trip. When an adapter is configured, pass an external vector to probe the adapter path as well:

```bash
V=$(curl -s http://localhost:8767/embed/single \
  -H 'Content-Type: application/json' \
  -d '{"text":"Water vapor condenses into clouds before precipitating as rain."}' \
  | jq '.vector')

curl -s http://localhost:8766/selftest \
  -H 'Content-Type: application/json' \
  -d '{"text":"Water vapor condenses into clouds before precipitating as rain.","vector":'"$V"',"steps":1}' \
  | jq
```

## Decoding External Vectors

Set `VEC2TEXT_PROCRUSTES_PATH` to the trained adapter before launching the server. The `/decode` endpoint will transform incoming vectors automatically when `apply_adapter` is true (default):

```bash
curl -s http://localhost:8766/decode \
  -H 'Content-Type: application/json' \
  -d '{"vectors":['"$V"'],"steps":1}' \
  | jq
```

Disable adaptation for teacher-space vectors:

```bash
curl -s http://localhost:8766/decode \
  -H 'Content-Type: application/json' \
  -d '{"vectors":['"$V"'],"steps":1,"apply_adapter":false}' \
  | jq
```
