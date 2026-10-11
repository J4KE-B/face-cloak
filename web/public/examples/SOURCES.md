# Example victims (live demo)

Official NASA astronaut portraits: works of the US federal government, public domain (17 U.S.C. 105). Every
Wikimedia Commons file page below carries `{{PD-USGov-NASA}}` and the licence metadata "Public domain" (checked via
the Commons API on 2026-10-11). Images were only downscaled (longest side 1024 px) and, for `*_thumb.jpg`, cropped
to the face. `*_a` is the photo that gets cloaked and swapped; `*_b` is a different photo of the same person used as
the judge's reference.

Pairs were chosen with `scripts/select_demo_examples.py` (14 candidates x 7 targets; demo defaults eps 8/255,
30 steps, fast mode; judge glintr100, tau 0.156). Judge similarity to the `_b` reference:

| id | Name | Target | Clean swap | Cloaked swap | `_a` source | `_b` source |
|----|------|--------|-----------:|-------------:|-------------|-------------|
| chawla | Kalpana Chawla | jemison | 0.845 | 0.006 | https://commons.wikimedia.org/wiki/File:Kalpana_Chawla,_official_portrait.jpg | https://commons.wikimedia.org/wiki/File:Kalpana_Chawla,_NASA_photo_portrait_in_orange_suit.jpg |
| ochoa | Ellen Ochoa | kim | 0.753 | -0.388 | https://commons.wikimedia.org/wiki/File:Ellen_Ochoa,_official_portrait.jpg | https://commons.wikimedia.org/wiki/File:Ellen_Ochoa.jpg |
| menon | Anil Menon | kim | 0.841 | -0.226 | https://commons.wikimedia.org/wiki/File:NASA_Astronaut_Anil_Menon_(jsc2024e015919).jpg | https://commons.wikimedia.org/wiki/File:NASA_Astronaut_Anil_Menon_(jsc2024e015918).jpg |
| chari | Raja Chari | jemison | 0.851 | -0.372 | https://commons.wikimedia.org/wiki/File:Raja_Chari_official_portrait.jpg | https://commons.wikimedia.org/wiki/File:Raja_Chari,_official_portrait_(2021).jpg |

`*_thumb.jpg` is a face crop of the matching `*_a.jpg`.

Caveats: numbers are from one CPU run on the downscaled JPEGs, without the browser re-encode the live site applies,
and PGD uses a random start, so live runs differ slightly. For Menon (consecutive frames of one 2024 shoot) and Chawla
(similar era) the `_a`/`_b` photos are not fully independent, so their clean-swap similarity is somewhat inflated.
