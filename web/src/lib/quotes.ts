export type Category = "wisdom" | "code" | "gaming" | "science" | "grit";

export interface Quote {
  id: string;
  text: string;
  author: string;
  source?: string;
  category: Category;
  custom?: boolean;
}

export interface CategoryMeta {
  id: Category;
  label: string;
  hero: string;
  /** Main accent used for the sprite, badges and the dialogue box glow. */
  color: string;
}

export const CATEGORIES: CategoryMeta[] = [
  { id: "wisdom", label: "Wisdom", hero: "The Sage", color: "#a78bfa" },
  { id: "code", label: "Code", hero: "Bit-Bot", color: "#4ade80" },
  { id: "gaming", label: "Gaming", hero: "Invader", color: "#f472b6" },
  { id: "science", label: "Science", hero: "Professor", color: "#38bdf8" },
  { id: "grit", label: "Grit", hero: "The Knight", color: "#fb923c" },
];

export const CATEGORY_BY_ID = Object.fromEntries(
  CATEGORIES.map((c) => [c.id, c]),
) as Record<Category, CategoryMeta>;

type RawQuote = [text: string, author: string, source?: string];

const RAW: Record<Category, RawQuote[]> = {
  wisdom: [
    [
      "Waste no more time arguing what a good man should be. Be one.",
      "Marcus Aurelius",
      "Meditations",
    ],
    [
      "It is not that we have a short time to live, but that we waste a lot of it.",
      "Seneca",
      "On the Shortness of Life",
    ],
    [
      "He who has a why to live can bear almost any how.",
      "Friedrich Nietzsche",
    ],
    ["The unexamined life is not worth living.", "Socrates", "Plato's Apology"],
    [
      "Life can only be understood backwards; but it must be lived forwards.",
      "Søren Kierkegaard",
    ],
    [
      "Not everything that is faced can be changed, but nothing can be changed until it is faced.",
      "James Baldwin",
    ],
    [
      "We are what we repeatedly do. Excellence, then, is not an act, but a habit.",
      "Will Durant",
    ],
    [
      "The journey of a thousand miles begins with a single step.",
      "Lao Tzu",
      "Tao Te Ching",
    ],
    [
      "Knowing others is intelligence; knowing yourself is true wisdom.",
      "Lao Tzu",
      "Tao Te Ching",
    ],
    ["The obstacle is the way.", "Ryan Holiday"],
  ],
  code: [
    [
      "Programs must be written for people to read, and only incidentally for machines to execute.",
      "Harold Abelson",
      "SICP",
    ],
    ["Premature optimization is the root of all evil.", "Donald Knuth"],
    ["Talk is cheap. Show me the code.", "Linus Torvalds"],
    ["Simplicity is prerequisite for reliability.", "Edsger W. Dijkstra"],
    [
      "Any fool can write code that a computer can understand. Good programmers write code that humans can understand.",
      "Martin Fowler",
    ],
    [
      "There are only two hard things in Computer Science: cache invalidation and naming things.",
      "Phil Karlton",
    ],
    ["Make it work, make it right, make it fast.", "Kent Beck"],
    [
      "Debugging is twice as hard as writing the code in the first place.",
      "Brian Kernighan",
    ],
    ["The best way to predict the future is to invent it.", "Alan Kay"],
    [
      "Code is like humor. When you have to explain it, it's bad.",
      "Cory House",
    ],
  ],
  gaming: [
    [
      "It's dangerous to go alone! Take this.",
      "Old Man",
      "The Legend of Zelda",
    ],
    [
      "Thank you Mario! But our princess is in another castle!",
      "Toad",
      "Super Mario Bros.",
    ],
    ["War. War never changes.", "Narrator", "Fallout"],
    [
      "The right man in the wrong place can make all the difference in the world.",
      "G-Man",
      "Half-Life 2",
    ],
    [
      "What is a man? A miserable little pile of secrets!",
      "Dracula",
      "Castlevania: Symphony of the Night",
    ],
    ["Stay awhile and listen.", "Deckard Cain", "Diablo"],
    ["Do a barrel roll!", "Peppy Hare", "Star Fox 64"],
    ["A man chooses, a slave obeys.", "Andrew Ryan", "BioShock"],
    [
      "Nothing is true, everything is permitted.",
      "Ezio Auditore",
      "Assassin's Creed II",
    ],
    ["The cake is a lie.", "Test subject graffiti", "Portal"],
  ],
  science: [
    [
      "The first principle is that you must not fool yourself, and you are the easiest person to fool.",
      "Richard Feynman",
    ],
    [
      "Nothing in life is to be feared, it is only to be understood.",
      "Marie Curie",
    ],
    ["All models are wrong, but some are useful.", "George E. P. Box"],
    [
      "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
      "Alan Turing",
      "Computing Machinery and Intelligence",
    ],
    ["The important thing is not to stop questioning.", "Albert Einstein"],
    [
      "If I have seen further it is by standing on the shoulders of Giants.",
      "Isaac Newton",
    ],
    [
      "The good thing about science is that it's true whether or not you believe in it.",
      "Neil deGrasse Tyson",
    ],
    ["Torture the data, and it will confess to anything.", "Ronald Coase"],
    ["Imagination is more important than knowledge.", "Albert Einstein"],
    [
      "Science is a way of thinking much more than it is a body of knowledge.",
      "Carl Sagan",
    ],
  ],
  grit: [
    ["Do. Or do not. There is no try.", "Yoda", "The Empire Strikes Back"],
    ["Fall seven times, stand up eight.", "Japanese proverb"],
    ["You miss 100% of the shots you don't take.", "Wayne Gretzky"],
    [
      "Ever tried. Ever failed. No matter. Try again. Fail again. Fail better.",
      "Samuel Beckett",
      "Worstward Ho",
    ],
    ["Hard work beats talent when talent doesn't work hard.", "Tim Notke"],
    ["Start where you are. Use what you have. Do what you can.", "Arthur Ashe"],
    ["Fortune favors the bold.", "Virgil", "Aeneid"],
    [
      "It does not matter how slowly you go as long as you do not stop.",
      "Confucius",
    ],
    [
      "What you do makes a difference, and you have to decide what kind of difference you want to make.",
      "Jane Goodall",
    ],
    [
      "A ship in harbor is safe, but that is not what ships are built for.",
      "John A. Shedd",
    ],
  ],
};

export const QUOTES: Quote[] = (Object.keys(RAW) as Category[]).flatMap(
  (category) =>
    RAW[category].map(([text, author, source], i) => ({
      id: `${category}-${i}`,
      text,
      author,
      source,
      category,
    })),
);

export function formatQuote(q: Pick<Quote, "text" | "author" | "source">) {
  return `"${q.text}" — ${q.author}${q.source ? `, ${q.source}` : ""}`;
}
