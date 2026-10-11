/**
 * Built-in example victims: public-domain NASA portraits (Wikimedia Commons, PD-USGov-NASA), see
 * public/examples/SOURCES.md. Each is paired with the bundled target where the clean swap most clearly shows the
 * victim; picked and verified (clean swap well above tau, cloaked swap below it) by scripts/select_demo_examples.py.
 */
export interface Example {
  id: string;
  name: string;
  /** Photo that gets cloaked and swapped. */
  photo: string;
  /** Second, different photo of the same person: the judge's reference. */
  reference: string;
  thumb: string;
  /** Recommended target id (must exist in the backend's bundled targets). */
  target: string;
  source: string;
}

const ex = (id: string, name: string, target: string, source: string): Example => ({
  id,
  name,
  photo: `/examples/${id}_a.jpg`,
  reference: `/examples/${id}_b.jpg`,
  thumb: `/examples/${id}_thumb.jpg`,
  target,
  source,
});

export const EXAMPLES: Example[] = [
  ex("ochoa", "Ellen Ochoa", "kim", "https://commons.wikimedia.org/wiki/File:Ellen_Ochoa,_official_portrait.jpg"),
  ex("chari", "Raja Chari", "jemison", "https://commons.wikimedia.org/wiki/File:Raja_Chari_official_portrait.jpg"),
  ex("menon", "Anil Menon", "kim", "https://commons.wikimedia.org/wiki/File:NASA_Astronaut_Anil_Menon_(jsc2024e015919).jpg"),
  ex("chawla", "Kalpana Chawla", "jemison", "https://commons.wikimedia.org/wiki/File:Kalpana_Chawla,_official_portrait.jpg"),
];
