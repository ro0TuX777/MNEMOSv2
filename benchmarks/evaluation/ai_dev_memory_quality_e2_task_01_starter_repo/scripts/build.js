const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const required = [
  "src/index.html",
  "src/app.js",
  "src/logic.js",
  "src/styles.css"
];

for (const rel of required) {
  const full = path.join(root, rel);
  if (!fs.existsSync(full)) {
    throw new Error(`Missing required file: ${rel}`);
  }
}

const dist = path.join(root, "dist");
fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist, { recursive: true });
for (const rel of required) {
  fs.copyFileSync(path.join(root, rel), path.join(dist, path.basename(rel)));
}

console.log("Build check complete.");

