(() => {
  const demo = document.querySelector('[data-rdma-demo]');
  if (!demo) return;

  const steps = [...demo.querySelectorAll('[data-rdma-phase]')];
  const controls = demo.querySelector('.rdma-controls');
  const count = demo.querySelector('[data-rdma-count]');
  const announcement = demo.querySelector('[data-rdma-announcement]');
  const previous = demo.querySelector('[data-rdma-action="previous"]');
  const next = demo.querySelector('[data-rdma-action="next"]');
  const play = demo.querySelector('[data-rdma-action="play"]');
  const last = steps.length - 1;
  let current = 0;
  let playback;

  function stop() {
    window.clearInterval(playback);
    playback = undefined;
  }

  function show(index) {
    current = index;
    demo.dataset.phase = steps[index].dataset.rdmaPhase;
    steps.forEach((step, position) => {
      step.inert = position !== index;
    });
    previous.setAttribute('aria-disabled', index === 0);
    next.setAttribute('aria-disabled', index === last);
    play.textContent = index === last ? 'Replay' : 'Play';
    if (playback !== undefined) play.textContent = 'Pause';
    count.textContent = `Step ${index + 1} of ${steps.length}`;
    announcement.textContent = `. ${steps[index].textContent.trim()}`;
  }

  function stepBy(offset) {
    const index = current + offset;
    if (index < 0 || index > last) return;
    stop();
    delete demo.dataset.paused;
    show(index);
  }

  previous.addEventListener('click', () => stepBy(-1));
  next.addEventListener('click', () => stepBy(1));
  play.addEventListener('click', () => {
    if (playback !== undefined) {
      stop();
      demo.dataset.paused = '';
      show(current);
      return;
    }

    delete demo.dataset.paused;
    playback = window.setInterval(() => {
      const index = current + 1;
      if (index === last) stop();
      show(index);
    }, 5200);
    show(current === last ? 0 : current);
  });

  show(0);
  controls.hidden = false;
})();
