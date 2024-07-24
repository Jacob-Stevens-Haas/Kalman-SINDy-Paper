# Build paper:
1. Verify LaTeX distribution installed (e.g. TexLive, including all extras)

```
sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

1. `pip install ./images/gen_image
2. Copy experiments into ./images/gen_image/trials
3. Run ./images/gen_image/composite_plots.py, saving relevant images into ./images
4. Build main.tex
