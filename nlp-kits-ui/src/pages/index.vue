<script setup>
import { BigBlobBg } from "@/assets/bg/BigBlobBg.module.js";

import RouterCard from "@/components/RouterCard.vue";

onMounted(() => {
  const colorbg = new BigBlobBg({
    dom: "box",
    colors: ["#8381c5", "#EBDEA4", "#FFFFEF", "#F3C7D3", "#F0AAEA", "#B28BCA"],
    loop: true,
  });
});
</script>

<template>
  <div class="fill-height position-relative">
    <div
      id="box"
      class="position-absolute top-0 bottom-0 left-0 right-0"
      style="z-index: 1"
    ></div>
    <v-container
      class="position-absolute top-0 bottom-0 left-0 right-0"
      style="z-index: 10"
    >
      <v-responsive class="align-centerfill-height mx-auto" max-width="900">
        <v-img draggable="false" height="240" src="@/assets/logo.png" />
        <div class="text-center">
          <p class="text-h5 font-weight-light mb-2 select-none">Welcome to</p>
          <h1 class="text-h2 font-weight-bold select-none">NLP Kits</h1>
        </div>
        <div class="py-4" />
        <v-row>
          <v-col cols="6">
            <router-card
              title="Classification"
              subtitle="Classify text into categories."
              router="/classify"
            />
          </v-col>
        </v-row>
      </v-responsive>
    </v-container>
  </div>
</template>
