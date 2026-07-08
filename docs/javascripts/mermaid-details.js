// Mermaid diagrams inside a collapsed <details> (e.g. a `???` admonition) can
// render at zero width, because their container is display:none when Material
// runs Mermaid on load. Material's SVGs are responsive (width:100%; max-width),
// so they usually re-fit via CSS once shown — this nudges any JS-driven sizing
// to recompute the first time a <details> is expanded. Harmless if unneeded.
document$.subscribe(() => {
  document.querySelectorAll("details").forEach((det) => {
    if (det.dataset._mmBound) return;
    det.dataset._mmBound = "1";
    det.addEventListener("toggle", () => {
      if (det.open && det.querySelector(".mermaid")) {
        window.dispatchEvent(new Event("resize"));
      }
    });
  });
});
