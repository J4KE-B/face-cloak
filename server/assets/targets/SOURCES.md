# Face-swap target images

Targets for the live demo. All are works of the US federal government (NASA official astronaut portraits),
which are in the public domain (17 U.S.C. 105; NASA media usage guidelines). Each Wikimedia Commons file page
carries the `{{PD-USGov-NASA}}` licence template (checked via the Commons API on the retrieval date; the
"Public domain" licence metadata returned by `imageinfo/extmetadata` agrees). Images were downscaled and
cropped to a 512x512 head-and-shoulders JPEG (<= 60 KB each); no other edits.

Retrieval date: 2026-10-11.

| id | Name | Source (Commons file page) | Original NASA record | Licence |
|----|------|----------------------------|----------------------|---------|
| duke | Charles Duke | https://commons.wikimedia.org/wiki/File:Official_NASA_portrait_Charles_Moss_Duke_Jr.jpg | NASA (via Commons) | Public domain, PD-USGov-NASA |
| godwin | Linda Godwin | https://commons.wikimedia.org/wiki/File:Official_portrait_of_astronaut_Linda_M._Godwin.jpg | https://images.nasa.gov/details-S90-26968 | Public domain, PD-USGov-NASA |
| jemison | Mae Jemison | https://commons.wikimedia.org/wiki/File:Mae_Jemison_-_Official_portrait_of_1987_astronaut_candidate.jpg | https://images.nasa.gov/details-s87-45893 | Public domain, PD-USGov-NASA |
| kim | Jonny Kim | https://commons.wikimedia.org/wiki/File:Jonny_Kim_official_portrait.jpg | https://www.flickr.com/photos/nasa2explore/44764444431/ | Public domain, PD-USGov-NASA |
| moghbeli | Jasmin Moghbeli | https://commons.wikimedia.org/wiki/File:Jasmin_Moghbeli_official_portrait.jpg | https://www.flickr.com/photos/nasa2explore/42953213510/ | Public domain, PD-USGov-NASA |
| rubio | Frank Rubio | https://commons.wikimedia.org/wiki/File:Frank_Rubio_official_portrait.jpg | https://www.flickr.com/photos/nasa2explore/44763354501/ | Public domain, PD-USGov-NASA |
| collins | Eileen Collins | generated at startup from `skimage.data.astronaut()` (not stored in the repo) | NASA | Public domain (NASA) |

Metadata used by the server lives in `targets.json`.
