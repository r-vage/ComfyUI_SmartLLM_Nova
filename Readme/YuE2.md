# YuE2 song preparation

Select **YuE2 Music** in Smart LM Loader with a text-capable model. Enter a song
concept, genre, mood and desired lyric language in `user_prompt`. The task returns
one JSON object containing `style` and `lyrics` strings. A concept or preceding
story produces a complete original song with sections, rhyme and repeated hooks.
Style describes genre, instruments, vocals, language, tempo and arrangement in
free text. It does not require MiniMax headings or include ABC notation.

To write lyrics first, enable multi-task mode and select **Song Lyrics** followed
by **YuE2 Music**. Structured lyric sheets also select conversion automatically, even
when YuE2 Music runs alone. Conversion asks the model only for style; code assembles
the final JSON using the retained lyrics. Training has separate composition and
conversion examples in both bundled variants and can be enabled or disabled.

Existing YuE2 `style`/`lyrics` JSON and MiniMax `caption`/`lyrics` JSON are accepted
as conversion sources. Decoded lyrics retain their exact Unicode, punctuation,
whitespace and line order. Musical descriptions must be nonempty strings.
Enclosing JSON fences are accepted. Missing, extra, duplicate and non-string
fields are rejected.

For Song Lyrics sheets, the shared Music 3 parser normalizes recognized section
labels and section spacing, retains sung lines, and removes the title, production
header and final `Structure:` summary from lyrics. Those directions and section
performance hints remain available to the model for style. A counted reference
such as `[Chorus repeat twice]` expands only when one distinct earlier Chorus
exists. Unknown directions, ambiguous repeats, malformed sheets and empty Song
Lyrics output fail clearly. The model cannot replace retained lyrics; supplying
lyrics during conversion triggers a corrective retry. The selected mode remains
fixed through that retry, and prompt/Training state is restored afterward.

See the [splitter tour](Visual_Tour.md#split-song-json) for the resizable JSON editor.

## Splitter and native wiring

Build these connections in your own ComfyUI audio graph. SmartLLM does not ship
a YuE2 workflow JSON. Use a ComfyUI release that provides the native YuE2 nodes.

Connect Smart LM Loader's text/result output to
**Split YuE2 [SmartLLM Nova]** → `song_json`. Connect output 0, **style**, to `style`
on **both YuE2 Generate ABC and YuE2 Generate Music**. Connect output 1,
**lyrics**, to `lyrics` on both generators.

The splitter accepts exactly `style` and `lyrics` strings, including an enclosing
JSON fence, and forwards their decoded values unchanged. It rejects an empty
style. Dynamic prompt expansion is disabled so literal braces in lyrics survive.
MiniMax JSON must pass through the YuE2 Music task before reaching this splitter.

ABC planning stays native and optional. Connect the native ABC output to the
music generator's `abc` input, or supply an empty string to disable planning.
SmartLLM does not generate ABC.

## Audio generation

Select your text-capable LLM and generation settings in Smart LM Loader. Load a
compatible YuE2 checkpoint for the native audio nodes. Connect the music
generator's `CONDITIONING` output to your sampling path and its `seconds` output
to **Empty YuE2 Latent Audio** → `seconds`, so latent length follows the generated
audio. Decode the sampled audio with the checkpoint's VAE and connect it to an
audio save node.

Set `max_duration` on **YuE2 Generate Music** for your intended duration. A value
such as 60 seconds is an editable test cap, not the model limit or a guaranteed
song length. Increase it for longer songs.

Explicit instrumental requests allow empty lyrics with instrumental/no-vocals
style direction. **Instrumental audio quality remains unverified.** Mocked
generation and workflow checks establish text handling and wiring, not musical
quality, pronunciation or live LLM reliability.
