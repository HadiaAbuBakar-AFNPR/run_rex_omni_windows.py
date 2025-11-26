from rex_omni import RexOmniWrapper, RexOmniVisualize

rex = RexOmniWrapper(
    model_path="IDEA-Research/Rex-Omni",
    backend="transformers",
    attn_implementation="eager",   # Required for Windows
    max_tokens=2048,
    temperature=0.0,
    top_p=0.05,
    top_k=1,
    repetition_penalty=1.05,
)

print("Model loaded successfully in eager mode on Windows!")
