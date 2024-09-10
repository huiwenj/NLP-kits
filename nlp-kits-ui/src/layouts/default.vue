<script setup>
import { useRoute, useRouter } from "vue-router";

const show = ref(false);

const $route = useRoute();
const $router = useRouter();

watchEffect(() => {
  show.value = $route.path !== "/";
});
</script>
<template>
  <v-theme-provider theme="light" with-background>
    <v-app>
      <v-toolbar v-show="show" color="#8381C5">
        <v-responsive class="mx-auto" max-width="900">
          <v-btn @click="$router.push('/')">
            <v-icon icon="mdi-arrow-left" start></v-icon>
            Back
          </v-btn>
        </v-responsive>
      </v-toolbar>
      <v-main>
        <v-container>
          <v-responsive class="mx-auto py-3" max-width="900">
            <router-view />
          </v-responsive>
        </v-container>
      </v-main>
    </v-app>
  </v-theme-provider>
</template>
