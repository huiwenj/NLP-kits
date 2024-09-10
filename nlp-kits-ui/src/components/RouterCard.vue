<script setup>
import { useRouter } from "vue-router";

defineComponent({
  name: "RouterCard",
});

const props = defineProps({
  title: String,
  subtitle: String,
  router: String,
});

const router = useRouter();
</script>

<template>
  <v-card
    append-icon="mdi-open-in-new"
    class="py-4 bg-blur-5"
    prepend-icon="mdi-text-box-outline"
    style="background-color: rgba(255, 255, 255, 0.35)"
    rounded="lg"
    :subtitle="props.subtitle"
    :title="props.title"
    variant="text"
    @click="router.push(props.router)"
  >
    <v-overlay opacity=".08" scrim="primary" contained model-value persistent />
  </v-card>
</template>

<style scoped></style>
