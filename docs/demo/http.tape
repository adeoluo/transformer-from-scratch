# Records docs/demo/http.gif from a real run: `vhs docs/demo/http.tape` from the repo root.
Output docs/demo/http.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Type `python -m transformer_from_scratch serve --port 18332 >/dev/null 2>&1 &` Enter
Sleep 2s
Sleep 136s
Type `clear` Enter
Sleep 1s
Show
Type `curl -s localhost:18332/health; echo` Enter
Sleep 1.5s
Type `curl -s localhost:18332/generate -d '{"prompt": [1, 3, 2, 0], "max_new_tokens": 5}' | jq -c` Enter
Sleep 3s
Type `curl -s localhost:18332/evaluate | jq -c` Enter
Sleep 3s
Hide
Type `kill %1` Enter
Sleep 1s
