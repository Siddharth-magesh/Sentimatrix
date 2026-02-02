// Sentimatrix Documentation Extra JavaScript

document.addEventListener('DOMContentLoaded', function() {
  // Copy button feedback enhancement
  document.querySelectorAll('.md-clipboard').forEach(function(button) {
    button.addEventListener('click', function() {
      const originalTitle = this.getAttribute('title');
      this.setAttribute('title', 'Copied!');
      setTimeout(() => {
        this.setAttribute('title', originalTitle);
      }, 2000);
    });
  });

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add external link indicators
  document.querySelectorAll('a[href^="http"]').forEach(link => {
    if (!link.hostname.includes('sentimatrix')) {
      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');
    }
  });

  // Version selector enhancements
  const versionSelector = document.querySelector('.md-version');
  if (versionSelector) {
    versionSelector.addEventListener('change', function(e) {
      const version = e.target.value;
      const currentPath = window.location.pathname;
      window.location.href = currentPath.replace(/\/[^\/]+\//, '/' + version + '/');
    });
  }

  // Code block language labels
  document.querySelectorAll('pre code').forEach(block => {
    const langClass = Array.from(block.classList).find(c => c.startsWith('language-'));
    if (langClass) {
      const lang = langClass.replace('language-', '');
      const wrapper = block.closest('.highlight');
      if (wrapper && !wrapper.querySelector('.code-lang')) {
        const label = document.createElement('span');
        label.className = 'code-lang';
        label.textContent = lang.toUpperCase();
        wrapper.insertBefore(label, wrapper.firstChild);
      }
    }
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', function(e) {
    // Press '/' to focus search
    if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
      const searchInput = document.querySelector('.md-search__input');
      if (searchInput && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
      }
    }

    // Press 'Escape' to close search
    if (e.key === 'Escape') {
      const searchInput = document.querySelector('.md-search__input');
      if (searchInput && document.activeElement === searchInput) {
        searchInput.blur();
        document.querySelector('.md-search__form').reset();
      }
    }
  });

  // Table of contents highlighting
  const observerOptions = {
    root: null,
    rootMargin: '-20% 0px -35% 0px',
    threshold: 0
  };

  const headings = document.querySelectorAll('h2[id], h3[id]');
  const tocLinks = document.querySelectorAll('.md-nav__link');

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        tocLinks.forEach(link => {
          link.classList.toggle('md-nav__link--active',
            link.getAttribute('href') === '#' + id);
        });
      }
    });
  }, observerOptions);

  headings.forEach(heading => observer.observe(heading));

  // Feature cards animation on scroll
  const cards = document.querySelectorAll('.card');
  const cardObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.1 });

  cards.forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    cardObserver.observe(card);
  });

  // Stats counter animation
  const stats = document.querySelectorAll('.stat-number');
  const statsObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const target = entry.target;
        const endValue = parseInt(target.textContent);
        if (!isNaN(endValue) && !target.classList.contains('counted')) {
          target.classList.add('counted');
          animateCounter(target, 0, endValue, 1500);
        }
      }
    });
  }, { threshold: 0.5 });

  stats.forEach(stat => statsObserver.observe(stat));

  function animateCounter(element, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const current = Math.round(start + range * easeOut);

      element.textContent = current + (element.dataset.suffix || '');

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }

    requestAnimationFrame(update);
  }
});

// Console branding
console.log('%cSentimatrix', 'font-size: 24px; font-weight: bold; color: #6200ea;');
console.log('%cAdvanced Sentiment Analysis Toolkit', 'font-size: 14px; color: #666;');
console.log('%cDocs: https://sentimatrix.dev', 'font-size: 12px; color: #888;');
