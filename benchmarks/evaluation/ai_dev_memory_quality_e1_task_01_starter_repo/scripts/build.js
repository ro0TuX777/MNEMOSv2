const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const src = path.join(root, "src");
const dist = path.join(root, "dist");
const required = ["index.html", "app.js", "logic.js", "styles.css"];

if (!fs.existsSync(src)) {
  throw new Error("src/ is missing");
}

for (const file of required) {
  if (!fs.existsSync(path.join(src, file))) {
    throw new Error("Missing src file: " + file);
  }
}

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist, { recursive: true });

for (const file of required) {
  fs.copyFileSync(path.join(src, file), path.join(dist, file));
}

console.log("Build completed");
