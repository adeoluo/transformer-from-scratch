# Records docs/demo/attention.gif from a real run: `vhs docs/demo/attention.tape` from the repo root.
Output docs/demo/attention.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 520
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Sleep 10s
Type `clear` Enter
Sleep 1s
Show
Type `python -m transformer_from_scratch attention --steps 150` Enter
Sleep 75s
