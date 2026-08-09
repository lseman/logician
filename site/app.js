// ── Scroll-triggered reveals ──
const obs = new IntersectionObserver(
	entries => {
		entries.forEach(e => {
			if (e.isIntersecting) {
				e.target.classList.add("up");
				obs.unobserve(e.target);
			}
		});
	},
	{ threshold: 0.1 },
);
document.querySelectorAll(".reveal").forEach(el => obs.observe(el));

// Immediately reveal hero elements
setTimeout(() => {
	document
		.querySelectorAll(".hero .reveal")
		.forEach(el => el.classList.add("up"));
}, 60);

// Nav shadow on scroll
const nav = document.getElementById("main-nav");
window.addEventListener(
	"scroll",
	() => {
		nav.classList.toggle("scrolled", window.scrollY > 10);
	},
	{ passive: true },
);

// ── Copy code button ──
function _copyCode(btn) {
	const code = btn.closest(".code-block").querySelector("code").textContent;
	navigator.clipboard.writeText(code).then(() => {
		btn.textContent = "copied!";
		setTimeout(() => {
			btn.textContent = "copy";
		}, 1500);
	});
}
