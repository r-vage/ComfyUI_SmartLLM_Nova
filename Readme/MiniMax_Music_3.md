# MiniMax Music 3

SmartLLM prepares musical direction and lyrics for MiniMax Music 3. The
**MiniMax Music 3** task generates song JSON; **Split MiniMax Music 3** converts
that JSON into separate `caption` and `lyrics` strings for ComfyUI's native
encoder.

## Generate a song description and lyrics

Select **MiniMax Music 3** on Smart LM Loader with a text-capable LLM, then enter
a song concept or existing lyrics in `user_prompt`. Florence does not support
this text-generation task. Use **Training** to include bundled examples.

For example: “Write a hopeful folk song about coming home, with German lyrics,
an intimate alto voice and fingerpicked guitar at 88 BPM.” To adapt a lyric
sheet, supply it directly or enable **Multi-Task**, choose **Song Lyrics** first
and **MiniMax Music 3** second.

For a style and subject, Music 3 writes the song itself using the same craft as
Song Lyrics: developed verses, pre-choruses, a repeated chorus hook, a bridge,
an instrumental break, final chorus and outro. It uses rhyme and singable line
lengths, and writes out repeated sections. You do not need to supply lyric lines.

The task selects its behavior automatically, without another widget:

| Input | Behavior |
| --- | --- |
| A concept, standalone or from a preceding story task | Compose caption and complete lyrics |
| Immediately after Song Lyrics | Convert the sheet, retaining its sung words |
| An existing structured lyric sheet or caption/lyrics JSON | Convert, retaining its sung words |

During conversion, Python extracts and retains the lyrics. The model receives
the entire source for musical context and generates only the caption; SmartLLM
assembles the final caption/lyrics object. The model cannot replace the sung
words, even if asked to rewrite or translate them. Make such changes before
Music 3. This guarantee begins with the input Music 3 receives, after any cleanup
by the preceding task. Supplied JSON retains the decoded `lyrics` string exactly.

For structured sheets, SmartLLM excludes the title, production header and final
`Structure:` summary, normalizes recognized section labels, and keeps sung words,
punctuation, language, line order and section order. Performance hints inside
section labels guide the caption. Section spacing is normalized. Counted repeat
references such as `[Chorus repeat twice]` or `[Chorus x2]` add two copies of a
previously defined chorus. If multiple choruses have different words, use a unique
numbered section reference. Missing sections, unclear counts and standalone
directions such as `(repeat)` or `(whisper softly)` produce parsing errors; write
out the lyrics or put performance hints inside a section label instead.

An unstructured standalone request defaults to composition. Unusable Song Lyrics
output and recognizable but malformed sheets or song JSON produce an error
before generation. Caption language defaults to English; composed lyrics follow
the requested language. Explicit instrumental requests use empty lyrics.
**Training** controls the separate composition and conversion examples. A
content-free diagnostic identifies the selected mode (`compose` or `convert`).
Review generated captions and composed lyrics for musical quality.

## Output format

The STRING result contains exactly two fields:

```json
{
  "caption": "Global Metadata: Instrumental ambient, serene and spacious.\n\nVocal Details: No vocals.\n\nArrangement: Sparse piano opens, warm strings build, then fade.",
  "lyrics": ""
}
```

The structured caption and separation of musical directions from sung text follow
the [official MiniMax Music 3 guidance](https://github.com/MiniMax-AI/MiniMax-Music3).
For vocal songs, `lyrics` holds sung words beneath tags such as `[Verse]` and
`[Chorus]`.

SmartLLM checks both fields are strings and that the nonempty caption begins with
`Global Metadata:`, followed by `Vocal Details:` and `Arrangement:` on their own
lines. Invalid generation, including vocal output with missing lyrics, gets one
corrective retry before reporting an error. Unicode, quotes, newlines and section
tags are preserved.

## Connect the encoder

Build these connections in your own ComfyUI audio graph. SmartLLM does not ship
a MiniMax Music 3 workflow JSON.

Choose either route with a ComfyUI release providing `MiniMaxMusic3TextEncode`.

### Splitter with the native encoder

Add **🌒 SmartLLM Nova → Conversion → Split MiniMax Music 3** (ID
`Split MiniMax Music 3 [SmartLLM Nova]`). Connect Smart LM Loader's **STRING result**
to `song_json`, then connect the splitter's `caption` and `lyrics` outputs to
the matching inputs on **MiniMax Music3 Text Encode**. You can also paste JSON
into the resizable **Song JSON** panel. See the [splitter tour](Visual_Tour.md#split-song-json).

The native encoder handles the MiniMax Music 3 CLIP model and generation controls.
Its `CONDITIONING` output feeds the sampling path, and `seconds` supplies the audio
latent duration.

The splitter validates the structure locally and preserves the decoded strings
exactly, including Unicode, quotes, line breaks and section tags. It accepts
enclosing JSON fences and empty instrumental lyrics. Invalid JSON, missing,
extra or duplicate fields, non-string values and invalid caption headings produce
errors without quoting the supplied content. Dynamic prompt expansion is disabled
for `song_json`, so literal text such as `{rain|sun}` is preserved.

The splitter only validates and separates text; it does no generation or
corrective retries. Smart Model Loader is optional for this route.

### Encoder wrapper

Install [Smart Model Loader Nova](https://github.com/r-vage/ComfyUI_SmartModelLoader_Nova)
and a ComfyUI release providing `MiniMaxMusic3TextEncode`. Add **Text Encode
MiniMax Music 3** from its Conditioning menu. Connect Smart LM Loader's **STRING result**
to `song_json`, and a MiniMax Music 3 CLIP to `clip`.

The wrapper parses and validates locally, then delegates to ComfyUI. It retains
the native `seed`, `max_duration`, `cfg_scale` and `top_k` controls, and returns
**CONDITIONING** then **seconds**. Dynamic prompt expansion is disabled for
`song_json`.

When switching between encoders, reconnect by socket name and carry over your
generation settings, since defaults may differ.
