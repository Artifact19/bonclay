/**
 * The faded monument photo behind each language card, and the credit its licence asks for.
 * Every photo is from Wikimedia Commons under CC BY, CC BY-SA, CC0 or the public domain. The files in
 * public/landing/monuments were scaled to 900 px wide and desaturated a little; nothing else was changed.
 * The monument is the one in the capital of the state where the language is most spoken.
 */

export type Monument = {
  /** Language code, the same as in LANGS. */
  code: string;
  name: string;
  city: string;
  /** Path under /public. */
  image: string;
  /** Plain-text name of the photographer, as given on the Commons file page. */
  author: string;
  /** Short licence name, e.g. "CC BY-SA 4.0". */
  license: string;
  licenseUrl: string;
  /** The Commons file page. */
  source: string;
};

const CC = {
  "BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
  "BY-SA 3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
  "BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
} as const;

const commons = (file: string) => `https://commons.wikimedia.org/wiki/File:${file}`;

export const MONUMENTS: Record<string, Monument> = {
  hi: {
    code: "hi",
    name: "India Gate",
    city: "Delhi",
    image: "/landing/monuments/hi.webp",
    author: "Subeesh Balan",
    license: "CC BY-SA 3.0",
    licenseUrl: CC["BY-SA 3.0"],
    source: commons("India_Gate_1_(New_Delhi).jpg"),
  },
  mr: {
    code: "mr",
    name: "Gateway of India",
    city: "Mumbai",
    image: "/landing/monuments/mr.webp",
    author: "SriSriChinmaya",
    license: "CC BY-SA 4.0",
    licenseUrl: CC["BY-SA 4.0"],
    source: commons("Gateway_of_India_in_the_evening,_Mumbai,_India.jpg"),
  },
  bn: {
    code: "bn",
    name: "Victoria Memorial",
    city: "Kolkata",
    image: "/landing/monuments/bn.webp",
    author: "Subhrajyoti07",
    license: "CC BY-SA 4.0",
    licenseUrl: CC["BY-SA 4.0"],
    source: commons("Victoria_Memorial_situated_in_Kolkata.jpg"),
  },
  gu: {
    code: "gu",
    name: "Swaminarayan Akshardham",
    city: "Gandhinagar",
    image: "/landing/monuments/gu.webp",
    author: "Harsh4101991",
    license: "CC BY-SA 3.0",
    licenseUrl: CC["BY-SA 3.0"],
    source: commons("Akshardham_Gandhinagar_Gujarat.jpg"),
  },
  pa: {
    code: "pa",
    name: "Open Hand Monument",
    city: "Chandigarh",
    image: "/landing/monuments/pa.webp",
    author: "Shanmugamp7",
    license: "CC BY-SA 3.0",
    licenseUrl: CC["BY-SA 3.0"],
    source: commons("Open_Hand_Monument_1.jpg"),
  },
  ta: {
    code: "ta",
    name: "Kapaleeshwarar Temple",
    city: "Chennai",
    image: "/landing/monuments/ta.webp",
    author: "PlaneMad",
    license: "CC BY-SA 3.0",
    licenseUrl: CC["BY-SA 3.0"],
    source: commons("Mylapore_Kapaleeshwarar_temple_facade.jpg"),
  },
  te: {
    code: "te",
    name: "Charminar",
    city: "Hyderabad",
    image: "/landing/monuments/te.webp",
    author: "Joydeep",
    license: "CC BY-SA 3.0",
    licenseUrl: CC["BY-SA 3.0"],
    source: commons("Charminar_Hyderabad_111.jpg"),
  },
  kn: {
    code: "kn",
    name: "Vidhana Soudha",
    city: "Bengaluru",
    image: "/landing/monuments/kn.webp",
    author: "Moheen Reeyad",
    license: "CC BY-SA 4.0",
    licenseUrl: CC["BY-SA 4.0"],
    source: commons("Vidhana_Soudha,_front_(01).jpg"),
  },
  ml: {
    code: "ml",
    name: "Padmanabhaswamy Temple",
    city: "Thiruvananthapuram",
    image: "/landing/monuments/ml.webp",
    author: "Shots by Karthik S.P",
    license: "CC BY 4.0",
    licenseUrl: CC["BY 4.0"],
    source: commons("Sree_Padmanabhaswamy_Temple,.jpg"),
  },
  or: {
    code: "or",
    name: "Lingaraja Temple",
    city: "Bhubaneswar",
    image: "/landing/monuments/or.webp",
    author: "Rajashree Talukdar",
    license: "CC BY-SA 4.0",
    licenseUrl: CC["BY-SA 4.0"],
    source: commons("Lingaraj_Temple_Complex_-_Odisha_-_X.jpg"),
  },
  as: {
    code: "as",
    name: "Kamakhya Temple",
    city: "Guwahati",
    image: "/landing/monuments/as.webp",
    author: "Kunal Dalui",
    license: "CC BY-SA 3.0",
    licenseUrl: CC["BY-SA 3.0"],
    source: commons("Kamakhya_Temple,_Guwahati.jpg"),
  },
};

/** The same data as a list, in card order, for the footer credits. */
export const MONUMENT_CREDITS: Monument[] = Object.values(MONUMENTS);
