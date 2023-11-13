function setupPW() {
  fixAccordion('tab_txt2img');  
  fixAccordion('tab_img2img');

  fixInputs('tab_txt2img', 'prompt');
  fixInputs('tab_img2img', 'neg_prompt');
}

function fixInputs(tab, type) {
  const elemId = `#${tab} #pw-${type}-prompt-slider`;
  const slider = document.querySelector(elemId);

  slider.querySelector('.head').remove();

  const newSpan = document.createElement("span");
  newSpan.innerHTML = `${type[0].toUpperCase() + type.slice(1)} Weight`;
  const ancestor = slider.parentNode.parentNode.parentNode;
  ancestor.insertBefore(newSpan, ancestor.firstChild);

  document.querySelector(`${elemId} + div input[type="number"]`).setAttribute("step", "0.01");
}

function fixAccordion(tab) {
  document.querySelector(`#${tab} #pw .icon`).remove();
  document.querySelector(`#${tab} #pw .open`).remove();
}

onUiLoaded(setupPW);