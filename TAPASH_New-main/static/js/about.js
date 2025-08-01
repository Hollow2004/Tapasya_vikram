let currentSection = 0;
const sections = document.querySelectorAll('section');
const totalSections = sections.length;

function scrollToSection(index) {
  sections[index].scrollIntoView({ behavior: 'smooth' });
}

function autoScroll() {
  currentSection = (currentSection + 1) % totalSections;
  scrollToSection(currentSection);
}

setInterval(autoScroll, 3000);

document.addEventListener('wheel', (event) => {
  if (event.deltaY > 0 && currentSection < totalSections - 1) {
    currentSection++;
    scrollToSection(currentSection);
  } else if (event.deltaY < 0 && currentSection > 0) {
    currentSection--;
    scrollToSection(currentSection);
  }
});