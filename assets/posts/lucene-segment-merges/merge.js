(() => {
  const figure = document.querySelector(".lucene-walkthrough");
  const steps = [...figure.querySelectorAll(".lucene-step")];
  const controls = figure.querySelector(".lucene-controls");
  const previous = controls.querySelector('[data-merge-action="previous"]');
  const next = controls.querySelector('[data-merge-action="next"]');
  const play = controls.querySelector('[data-merge-action="play"]');
  const count = controls.querySelector(".lucene-count");
  const last = steps.length - 1;
  let current = 0;
  let timer;

  function show(index) {
    current = index;
    steps.forEach((step, position) => {
      step.inert = position !== current;
    });
    previous.setAttribute("aria-disabled", current === 0);
    next.setAttribute("aria-disabled", current === last);
    count.textContent = `Step ${current + 1} of ${steps.length}`;
  }

  function pause() {
    window.clearInterval(timer);
    timer = undefined;
    play.textContent = current === last ? "Replay animation" : "Play animation";
  }

  controls.addEventListener("click", ({ target }) => {
    if (target.getAttribute("aria-disabled") === "true") return;
    switch (target.dataset.mergeAction) {
      case "previous":
        show(current - 1);
        pause();
        break;
      case "next":
        show(current + 1);
        pause();
        break;
      case "play":
        if (timer !== undefined) {
          pause();
          break;
        }
        if (current === last) {
          show(0);
        }
        play.textContent = "Pause animation";
        timer = window.setInterval(() => {
          show(current + 1);
          if (current === last) pause();
        }, 3200);
        break;
    }
  });

  show(0);
  figure.dataset.interactive = "";
  controls.hidden = false;
})();
