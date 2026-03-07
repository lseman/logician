const m = require("markdown-it")();
const html = m.render("# \x1b[36mLogician\x1b[0m CLI");
console.log(html);
